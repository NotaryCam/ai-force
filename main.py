from fastapi import FastAPI, UploadFile, File
from typing import List
import fitz  # PyMuPDF
import os
import base64
import requests
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
MODEL = "gpt-4o"

INPUT_COST_PER_1K = 0.0025
OUTPUT_COST_PER_1K = 0.01

app = FastAPI(
    title="PDF Form Classifier API",
    description="Detect Form3200 in uploaded PDF (including scanned pages) and return page range",
    version="3.1.0"
)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
}

sample_form_text = ""  # Global variable to store sample form text


def extract_pages_text(file_bytes: bytes) -> List[str]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return [page.get_text().strip() for page in doc]
    except Exception as e:
        raise ValueError(f"Error extracting PDF: {str(e)}")


def remove_footer(text: str, lines_to_strip: int = 2) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:-lines_to_strip]) if len(lines) > lines_to_strip else text


def page_to_base64_image(doc, page_index) -> str:
    try:
        pix = doc[page_index].get_pixmap(dpi=200)
        image_bytes = pix.tobytes("png")
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error converting page {page_index + 1} to image: {e}")


def count_tokens(messages, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for msg in messages:
        num_tokens += 4  # Base cost per message
        for key, value in msg.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "text" in item:
                        num_tokens += len(encoding.encode(item["text"]))
            elif isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            # Skip encoding if not string or handled
    num_tokens += 2  # Priming for assistant reply
    return num_tokens


@app.on_event("startup")
def load_sample_form():
    global sample_form_text
    try:
        with open("form3200_sample.pdf", "rb") as f:
            file_bytes = f.read()
        pages_text = extract_pages_text(file_bytes)
        pages_text = [remove_footer(text) for text in pages_text]
        sample_form_text = "\n".join(pages_text)
        print("✅ Sample form3200 loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading sample form: {e}")


@app.post("/classify")
async def classify_form(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {"error": str(e)}

    if not sample_form_text:
        return {"error": "Sample form not loaded. Please ensure 'form3200_sample.pdf' is present."}

    form_pages = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, page in enumerate(doc):
        raw_text = page.get_text().strip()
        text = remove_footer(raw_text)

        if text:
            messages = [
                {
                    "role": "system",
                    "content": "You are a document classifier. Only respond 'Yes' or 'No'."
                },
                {
                    "role": "user",
                    "content": f"""You are comparing form structures. Ignore differences in footers (such as page numbers or office names). The following is a sample Form3200 (footer removed):\n\n{sample_form_text}\n\nNow, does the following page (footer removed) from another document belong to Form3200?\n\n{text}"""
                }
            ]
        else:
            base64_image = page_to_base64_image(doc, i)
            messages = [
                {
                    "role": "system",
                    "content": "You are a document classifier. Only respond 'Yes' or 'No'."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Does this scanned page belong to Form3200? Ignore differences in footer (e.g., page numbers or office stamps)."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ]

        input_tokens = count_tokens(messages, MODEL)
        total_input_tokens += input_tokens

        payload = {"messages": messages}
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)

        if response.status_code != 200:
            return {
                "error": "Failed during classification",
                "page": i + 1,
                "details": response.text,
                "input_tokens_used": total_input_tokens,
                "output_tokens_used": total_output_tokens,
                "estimated_cost_usd": round(
                    (total_input_tokens / 1000) * INPUT_COST_PER_1K +
                    (total_output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
                )
            }

        reply = response.json()['choices'][0]['message']['content']
        output_tokens = count_tokens([response.json()['choices'][0]['message']], MODEL)
        total_output_tokens += output_tokens

        if reply.strip().lower() == 'yes':
            form_pages.append(i)

    # Final ranges
    if not form_pages:
        return {
            "form3200_found": False,
            "message": "Form3200 not found in the document.",
            "input_tokens_used": total_input_tokens,
            "output_tokens_used": total_output_tokens,
            "estimated_cost_usd": round(
                (total_input_tokens / 1000) * INPUT_COST_PER_1K +
                (total_output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
            )
        }

    ranges = []
    start = form_pages[0]
    end = start
    for i in range(1, len(form_pages)):
        if form_pages[i] == end + 1:
            end = form_pages[i]
        else:
            ranges.append((start + 1, end + 1))
            start = end = form_pages[i]
    ranges.append((start + 1, end + 1))

    return {
        "form3200_found": True,
        "page_ranges": [{"from": r[0], "to": r[1]} for r in ranges],
        "input_tokens_used": total_input_tokens,
        "output_tokens_used": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(
            (total_input_tokens / 1000) * INPUT_COST_PER_1K +
            (total_output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
        )
    }
