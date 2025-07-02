from fastapi import FastAPI, UploadFile, File
from typing import List, Dict, Any
import fitz  # PyMuPDF
import os
import base64
import requests
from dotenv import load_dotenv
import tiktoken
import json
import re
 
# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
MODEL = "gpt-4o"
 
# Debug: Print environment variables (remove in production)
print(f"🔍 AZURE_OPENAI_ENDPOINT loaded: {AZURE_OPENAI_ENDPOINT is not None}")
print(f"🔍 AZURE_OPENAI_API_KEY loaded: {AZURE_OPENAI_API_KEY is not None}")
if not AZURE_OPENAI_ENDPOINT:
    print("❌ AZURE_OPENAI_ENDPOINT is None - check your .env file")
if not AZURE_OPENAI_API_KEY:
    print("❌ AZURE_OPENAI_API_KEY is None - check your .env file")
 
INPUT_COST_PER_1K = 0.0025
OUTPUT_COST_PER_1K = 0.01
 
app = FastAPI(
    title="PDF Form Classifier & Extractor API",
    description="Detect Form3200 and extract filled-in field data",
    version="5.0.0"
)
 
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AZURE_OPENAI_API_KEY}",
}
 
sample_form_text = ""  # Global variable to store sample form text
sample_form_pages = []  # Store individual page texts for better comparison
 
 
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
        num_tokens += 4
        for key, value in msg.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "text" in item:
                        num_tokens += len(encoding.encode(item["text"]))
            elif isinstance(value, str):
                num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens
 
 
@app.on_event("startup")
def load_sample_form():
    global sample_form_text, sample_form_pages
   
    # Validate environment variables first
    if not AZURE_OPENAI_ENDPOINT:
        print("❌ AZURE_OPENAI_ENDPOINT not found. Please check your .env file.")
        print("Expected format: AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-15-preview")
        return
       
    if not AZURE_OPENAI_API_KEY:
        print("❌ AZURE_OPENAI_API_KEY not found. Please check your .env file.")
        return
   
    try:
        with open("form3200_sample.pdf", "rb") as f:
            file_bytes = f.read()
        pages_text = extract_pages_text(file_bytes)
        sample_form_pages = [remove_footer(text) for text in pages_text]
        sample_form_text = "\n".join(sample_form_pages)
        print("✅ Sample form3200 loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading sample form: {e}")
 
 
def get_enhanced_classification_prompt(page_text: str, page_number: int, total_pages: int) -> List[Dict[str, Any]]:
    """
    Enhanced prompt for Form 3200 classification with better context and structure
    """
   
    # Get the corresponding sample page if available
    sample_page_text = ""
    if page_number < len(sample_form_pages):
        sample_page_text = sample_form_pages[page_number]
   
    system_prompt = """You are a specialized document classification expert with expertise in Form 3200 (MULTISTATE FIXED RATE NOTE) identification. Your task is to determine whether a document page belongs to Form 3200 by analyzing its structure, content, and layout.
 
FORM 3200 IDENTIFICATION CRITERIA:
- EXACT form identifier: "MULTISTATE FIXED RATE NOTE—Single Family—Fannie Mae/Freddie Mac UNIFORM INSTRUMENT Form 3200"
- Must contain "Form 3200" in the header/footer
- NOT Form 3202 (Alaska), NOT Form 3210 (Florida), NOT any other form number
- Standard 3-page promissory note structure with sections 1-10
- Page 1: NOTE header, sections 1-5 (Promise to Pay, Interest, Payments, Prepayment, Loan Charges)
- Page 2: Sections 6-9 (Failure to Pay, Notices, Obligations, Waivers)  
- Page 3: Section 10 (Uniform Secured Note), signature area, "Sign Original Only"
 
CRITICAL CLASSIFICATION RULES:
- FIRST check for "Form 3200" identifier in headers/footers
- If form number is NOT 3200, answer "No" immediately
- If form number IS 3200, then check structural compatibility
- Accept both filled and blank Form 3200 templates
- Focus on section structure and legal language patterns
- Ignore differences between filled vs blank fields
- ALL THREE PAGES must be Form 3200 for complete document classification
 
RESPONSE FORMAT:
Respond with exactly one word: "Yes" or "No"
 
ANALYSIS SEQUENCE:
1. Search for "Form 3200" identifier first
2. Verify "MULTISTATE FIXED RATE NOTE" title
3. Check section numbering and structure
4. Confirm page position matches expected Form 3200 layout"""
 
    if sample_page_text:
        # Define expected content for each page
        expected_content = {
            0: "NOTE header, BORROWER'S PROMISE TO PAY, INTEREST section, PAYMENTS section, BORROWER'S RIGHT TO PREPAY, LOAN CHARGES",
            1: "BORROWER'S FAILURE TO PAY AS REQUIRED, GIVING OF NOTICES, OBLIGATIONS OF PERSONS UNDER THIS NOTE, WAIVERS",
            2: "UNIFORM SECURED NOTE section, signature lines with borrower names, WITNESS THE HAND(S) AND SEAL(S), Sign Original Only"
        }
       
        user_prompt = f"""REFERENCE FORM 3200 PAGE {page_number + 1}:
{sample_page_text}
 
DOCUMENT TO CLASSIFY (Page {page_number + 1} of {total_pages}):
{page_text}
 
TASK: Determine if this page belongs to Form 3200 (MULTISTATE FIXED RATE NOTE).
 
STEP 1 - FORM IDENTIFIER CHECK:
- Look for "Form 3200" in headers/footers
- Look for "MULTISTATE FIXED RATE NOTE" title
- If you find "Form 3202", "Form 3210", or any other form number, answer "No"
 
STEP 2 - STRUCTURAL VERIFICATION (only if Step 1 confirms Form 3200):
- Expected content for page {page_number + 1}: {expected_content.get(page_number, "Unknown page structure")}
- Compare section structure and numbering
- Verify legal language patterns match Form 3200 format
- Accept both blank templates and filled forms
 
IMPORTANT NOTES:
- A blank Form 3200 template is still Form 3200
- Focus on form number identification first, then structure
- Differences in filled vs blank fields should be ignored
- Only the form number and basic structure matter
 
Does this page belong to Form 3200 (not 3202, 3210, or any other form)?"""
    else:
        user_prompt = f"""DOCUMENT TO CLASSIFY (Page {page_number + 1} of {total_pages}):
{page_text}
 
TASK: Determine if this page belongs to Form 3200 (MULTISTATE FIXED RATE NOTE).
 
STEP 1 - FORM IDENTIFIER CHECK:
- Search for "Form 3200" in the page content
- Look for "MULTISTATE FIXED RATE NOTE" title
- Reject if you find "Form 3202", "Form 3210", or any other form number
 
STEP 2 - BASIC STRUCTURE CHECK (only if Step 1 confirms Form 3200):
- Verify this appears to be a promissory note
- Check for numbered sections (1-10)
- Confirm standard legal document formatting
 
CRITICAL: Answer "Yes" only if this is specifically Form 3200, not any other form number.
 
Does this page belong to Form 3200?"""
 
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
 
 
def get_enhanced_image_classification_prompt(page_number: int, total_pages: int) -> List[Dict[str, Any]]:
    """
    Enhanced prompt for image-based Form 3200 classification
    """
   
    system_prompt = """You are a specialized document classification expert trained to identify Form 3200 (MULTISTATE FIXED RATE NOTE) from scanned images. Your task is to analyze the visual layout, text patterns, and form identifiers to determine if this scanned page belongs to Form 3200.
 
FORM 3200 IDENTIFICATION PRIORITY:
1. FIRST: Look for "Form 3200" identifier in headers/footers
2. SECOND: Look for "MULTISTATE FIXED RATE NOTE" title
3. REJECT: Any form with "Form 3202", "Form 3210", or other form numbers
4. ACCEPT: Both blank templates and filled Form 3200 documents
 
VISUAL IDENTIFICATION MARKERS:
- Header/footer text: "MULTISTATE FIXED RATE NOTE—Single Family—Fannie Mae/Freddie Mac UNIFORM INSTRUMENT Form 3200"
- Form number "3200" clearly visible (not 3202, 3210, etc.)
- Standard promissory note layout with numbered sections 1-10
- Legal document formatting with section headers and subsections
 
RESPONSE FORMAT:
Respond with exactly one word: "Yes" or "No"
 
ANALYSIS PRIORITY:
1. Form number identification takes precedence over everything else
2. If form number is not 3200, answer "No" regardless of similarity
3. If form number is 3200, verify basic promissory note structure
4. Accept both blank forms and completed forms"""
 
    user_content = [
        {
            "type": "text",
            "text": f"""TASK: Analyze this scanned document page to determine if it belongs to Form 3200 (MULTISTATE FIXED RATE NOTE).
 
CONTEXT: This is page {page_number + 1} of {total_pages} in the document.
 
CRITICAL IDENTIFICATION STEPS:
1. FORM NUMBER CHECK: Look for "Form 3200" in headers/footers
   - If you see "Form 3202" (Alaska) → Answer "No"
   - If you see "Form 3210" (Florida) → Answer "No"
   - If you see "Form 3200" (Multistate) → Continue to step 2
   - If no form number visible → Continue to step 2
 
2. TITLE CHECK: Look for "MULTISTATE FIXED RATE NOTE"
   - If you see "ALASKA FIXED RATE NOTE" → Answer "No"
   - If you see "FLORIDA FIXED RATE NOTE" → Answer "No"
   - If you see "MULTISTATE FIXED RATE NOTE" → Answer "Yes"
 
3. STRUCTURE VERIFICATION (if steps 1-2 are unclear):
   - Standard promissory note sections 1-10
   - Legal document formatting
   - Fannie Mae/Freddie Mac uniform instrument layout
 
REMEMBER: Form number is the most important identifier. A blank Form 3200 template is still Form 3200.
 
Does this scanned page belong to Form 3200 (MULTISTATE, not Alaska 3202 or Florida 3210)?"""
        }
    ]
   
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
 
 
def get_enhanced_extraction_prompt(page_text: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Enhanced prompt for field extraction with better structure and examples
    """
   
    system_prompt = """You are a specialized form data extraction expert trained to identify and extract filled-in field values from Form 3200 (MULTISTATE FIXED RATE NOTE) documents. Your task is to locate all completed fields and return their values in a structured JSON format.
 
FORM 3200 FIELD TYPES:
- Date fields (Note Date, payment dates, maturity date)
- Location fields (City, State, Property Address)
- Financial amounts (Principal amount, interest rate, monthly payment)
- Personal information (Borrower names, lender information)
- Terms and conditions (payment schedules, late charges)
- Checkboxes and selections (payment methods, options)
 
EXTRACTION GUIDELINES:
- Only extract fields that contain actual filled-in data (not blank or template text)
- Preserve original formatting, spacing, and punctuation of extracted values
- Use clear, descriptive field names that reflect the form's structure
- Handle various input formats: typed text, handwritten content, printed data
- Ignore form structure elements like section headers, instructions, and blank fields
- Maintain data accuracy and completeness for legal document integrity
 
OUTPUT FORMAT:
Return valid JSON object with field names as keys and extracted values as strings.
 
FIELD NAMING CONVENTIONS:
- Use descriptive names: "Note_Date", "Principal_Amount", "Interest_Rate"
- Include page context for clarity: "Borrower_Name_1", "Borrower_Name_2"
- Preserve financial formatting: "$100,017.00" not "100017"
- Keep date formats as shown: "March 1, 2024" not "03/01/2024"
 
EXAMPLES FOR FORM 3200:
- "Note_Date": "March 1, 2024"
- "Property_City": "Oakland"
- "Property_State": "CA"
- "Principal_Amount": "$100,017.00"
- "Interest_Rate": "4.750%"
- "Monthly_Payment": "$1,000.17"
- "Maturity_Date": "April 1, 2049"
- "Borrower_1": "Joe Basallon"
- "Borrower_2": "Adnan Ashraf"""
 
    # Define expected content and fields for each page
    page_context = {
        0: {
            "sections": "Sections 1-5: Promise to Pay, Interest, Payments, Prepayment Rights, Loan Charges",
            "key_fields": "Note date, city, state, property address, principal amount, lender name, interest rate, monthly payment amount, maturity date"
        },
        1: {
            "sections": "Sections 6-9: Failure to Pay, Giving of Notices, Obligations, Waivers",
            "key_fields": "Late charge percentage, default terms, notice requirements, contact information"
        },
        2: {
            "sections": "Section 10: Uniform Secured Note, Signature Area",
            "key_fields": "Borrower signatures, borrower names, signature date, witness information"
        }
    }
   
    current_page_info = page_context.get(page_number, {"sections": "Unknown", "key_fields": "Various fields"})
   
    user_prompt = f"""FORM 3200 PAGE {page_number + 1} CONTENT:
{page_text}
 
PAGE CONTEXT: {current_page_info["sections"]}
EXPECTED FIELDS: {current_page_info["key_fields"]}
 
EXTRACTION TASK:
Analyze the Form 3200 page content above and extract all filled-in field values. Focus on:
 
1. SYSTEMATIC SCANNING: Read through the page content systematically
2. FIELD IDENTIFICATION: Locate all completed fields with actual data (ignore blank lines and template text)
3. VALUE EXTRACTION: Extract exact values as they appear, preserving formatting
4. DATA VALIDATION: Ensure extracted data makes sense in the context of a promissory note
 
SPECIFIC INSTRUCTIONS FOR PAGE {page_number + 1}:
{"- Extract note date, location, property address, principal amount, interest rate" if page_number == 0 else ""}
{"- Extract late charge percentages, grace periods, and any filled contact information" if page_number == 1 else ""}
{"- Extract borrower names from signature lines and any witness information" if page_number == 2 else ""}
- Look for handwritten, typed, or printed entries that fill in blank spaces
- Preserve exact formatting of amounts (include $ and commas), dates, and percentages
- Use descriptive JSON keys that clearly identify the field type and content
 
Return the extracted data as a clean JSON object with no additional text."""
 
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
 
 
def get_enhanced_image_extraction_prompt(page_number: int) -> List[Dict[str, Any]]:
    """
    Enhanced prompt for image-based field extraction
    """
   
    system_prompt = """You are an advanced OCR and form data extraction specialist trained to identify and extract filled-in field values from scanned Form 3200 documents. Your expertise includes reading handwritten text, typed content, and various form elements from scanned images.
 
EXTRACTION CAPABILITIES:
- Handwritten text recognition and interpretation
- Typed/printed text extraction
- Checkbox and selection identification
- Date and numerical value extraction
- Signature and mark recognition
- Field boundary detection and data association
 
QUALITY STANDARDS:
- Maintain high accuracy in text recognition
- Preserve original formatting and spacing
- Handle various writing styles and print qualities
- Distinguish between filled and empty fields
- Provide clear field-to-value associations
 
OUTPUT REQUIREMENTS:
- Return valid JSON object only
- Use descriptive field names as keys
- Include only fields with actual content
- Preserve original data formatting
- No additional text or explanations
 
FIELD RECOGNITION PRIORITIES:
1. Personal information (names, addresses, contacts)
2. Dates and numerical values
3. Checkbox selections and choices
4. Text areas and descriptions
5. Signatures and authorization marks"""
 
    user_content = [
        {
            "type": "text",
            "text": f"""SCANNED FORM 3200 PAGE {page_number + 1} EXTRACTION:
 
TASK REQUIREMENTS:
1. Carefully examine the entire scanned page for filled-in fields
2. Read and extract all handwritten, typed, or marked content
3. Identify field labels and associate them with their values
4. Distinguish between form structure and actual filled data
5. Handle various text orientations and writing qualities
 
EXTRACTION PROCESS:
- Scan systematically from top to bottom, left to right
- Look for completed text fields, checkboxes, and selection marks
- Extract dates, names, addresses, numbers, and other data
- Use logical field names based on form labels and context
- Ensure accuracy in text recognition and data association
 
OUTPUT: Return only the JSON object with extracted field data. Include only fields that contain actual filled-in content."""
        }
    ]
   
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
 
 
def classify_pages(doc) -> (bool, List[int], int, int):
    """
    Enhanced page classification with improved prompting and debugging
    """
    # Validate environment before proceeding
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError("Azure OpenAI configuration not found. Please check your .env file.")
   
    form_pages = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_pages = len(doc)
 
    print(f"🔍 Classifying {total_pages} pages...")
 
    for i, page in enumerate(doc):
        raw_text = page.get_text().strip()
        text = remove_footer(raw_text)
 
        # Debug: Check for form identifiers in the text
        if "Form 3200" in raw_text:
            print(f"📄 Page {i+1}: Found 'Form 3200' identifier")
        elif "Form 3202" in raw_text:
            print(f"📄 Page {i+1}: Found 'Form 3202' (Alaska) - should be rejected")
        elif "Form 3210" in raw_text:
            print(f"📄 Page {i+1}: Found 'Form 3210' (Florida) - should be rejected")
        elif "MULTISTATE FIXED RATE NOTE" in raw_text:
            print(f"📄 Page {i+1}: Found 'MULTISTATE FIXED RATE NOTE' title")
        else:
            print(f"📄 Page {i+1}: No clear form identifier found")
 
        if text:
            messages = get_enhanced_classification_prompt(text, i, total_pages)
        else:
            base64_image = page_to_base64_image(doc, i)
            messages = get_enhanced_image_classification_prompt(i, total_pages)
            # Add the image to the user message
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
 
        input_tokens = count_tokens(messages, MODEL)
        total_input_tokens += input_tokens
 
        payload = {"messages": messages}
       
        try:
            response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Azure OpenAI: {str(e)}")
 
        if response.status_code != 200:
            raise RuntimeError(f"Azure OpenAI API error: {response.status_code} - {response.text}")
 
        reply = response.json()['choices'][0]['message']['content']
        output_tokens = count_tokens([response.json()['choices'][0]['message']], MODEL)
        total_output_tokens += output_tokens
 
        # Debug: Show classification result
        print(f"🤖 Page {i+1} classification: {reply.strip()}")
 
        if reply.strip().lower() == 'yes':
            form_pages.append(i)
 
    print(f"✅ Classification complete. Form 3200 pages found: {form_pages}")
    return bool(form_pages), form_pages, total_input_tokens, total_output_tokens
 
 
@app.post("/classify")
async def classify_form(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {"error": str(e)}
 
    if not sample_form_text:
        return {"error": "Sample form not loaded. Please ensure 'form3200_sample.pdf' is present."}
 
    form3200_found, form_pages, input_tokens, output_tokens = classify_pages(doc)
 
    # Enhanced range detection for complete Form 3200 documents
    ranges = []
    complete_form3200_ranges = []
   
    if form_pages:
        # Basic contiguous ranges
        start = form_pages[0]
        end = start
        for i in range(1, len(form_pages)):
            if form_pages[i] == end + 1:
                end = form_pages[i]
            else:
                ranges.append((start + 1, end + 1))
                start = end = form_pages[i]
        ranges.append((start + 1, end + 1))
       
        # Identify complete 3-page Form 3200 documents
        for range_start, range_end in ranges:
            if range_end - range_start + 1 == 3:  # Exactly 3 pages
                complete_form3200_ranges.append({
                    "from": range_start,
                    "to": range_end,
                    "type": "complete_form3200",
                    "pages": 3
                })
            else:
                # Partial or extended ranges
                complete_form3200_ranges.append({
                    "from": range_start,
                    "to": range_end,
                    "type": "partial_or_extended",
                    "pages": range_end - range_start + 1,
                    "note": "May contain incomplete Form 3200 or additional pages"
                })
 
    return {
        "form3200_found": form3200_found,
        "total_pages_in_document": len(doc),
        "form3200_pages_detected": len(form_pages),
        "page_ranges": [{"from": r[0], "to": r[1]} for r in ranges],
        "detailed_ranges": complete_form3200_ranges,
        "summary": {
            "complete_form3200_documents": len([r for r in complete_form3200_ranges if r["type"] == "complete_form3200"]),
            "partial_matches": len([r for r in complete_form3200_ranges if r["type"] == "partial_or_extended"])
        },
        "input_tokens_used": input_tokens,
        "output_tokens_used": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": round(
            (input_tokens / 1000) * INPUT_COST_PER_1K +
            (output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
        )
    }
 
 
@app.post("/extract-fields")
async def extract_fields(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {"error": str(e)}
 
    merged_fields = {}
    total_input_tokens = 0
    total_output_tokens = 0
 
    for i, page in enumerate(doc):
        text = page.get_text().strip()
 
        if text:
            messages = get_enhanced_extraction_prompt(text, i)
        else:
            base64_image = page_to_base64_image(doc, i)
            messages = get_enhanced_image_extraction_prompt(i)
            # Add the image to the user message
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
 
        input_tokens = count_tokens(messages, MODEL)
        total_input_tokens += input_tokens
 
        payload = {"messages": messages}
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
 
        if response.status_code != 200:
            return {"error": f"Extraction failed on page {i + 1}", "details": response.text}
 
        reply = response.json()["choices"][0]["message"]["content"]
        output_tokens = count_tokens([response.json()["choices"][0]["message"]], MODEL)
        total_output_tokens += output_tokens
 
        try:
            # Enhanced JSON parsing with better error handling
            cleaned_reply = reply.strip()
            if cleaned_reply.startswith("```json"):
                cleaned_reply = cleaned_reply.removeprefix("```json").removesuffix("```").strip()
            elif cleaned_reply.startswith("```"):
                cleaned_reply = cleaned_reply.removeprefix("```").removesuffix("```").strip()
           
            # Additional cleaning for common formatting issues
            if cleaned_reply.startswith("JSON:"):
                cleaned_reply = cleaned_reply.removeprefix("JSON:").strip()
               
            fields = json.loads(cleaned_reply)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from the response
            try:
                json_match = re.search(r'\{.*\}', reply, re.DOTALL)
                if json_match:
                    fields = json.loads(json_match.group())
                else:
                    fields = {}
            except:
                fields = {}
 
        if isinstance(fields, dict):
            merged_fields.update(fields)
 
    return {
        "extracted_data": merged_fields,
        "input_tokens_used": total_input_tokens,
        "output_tokens_used": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(
            (total_input_tokens / 1000) * INPUT_COST_PER_1K +
            (total_output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
        )
    }
 
 
@app.post("/classify-and-extract")
async def detect_and_extract(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {"error": str(e)}
 
    if not sample_form_text:
        return {"error": "Sample form not loaded. Please ensure 'form3200_sample.pdf' is present."}
 
    form3200_found, _, input_tokens_c, output_tokens_c = classify_pages(doc)
 
    if not form3200_found:
        return {
            "form3200_found": False,
            "message": "The uploaded document is not Form3200.",
            "input_tokens_used": input_tokens_c,
            "output_tokens_used": output_tokens_c,
            "estimated_cost_usd": round(
                (input_tokens_c / 1000) * INPUT_COST_PER_1K +
                (output_tokens_c / 1000) * OUTPUT_COST_PER_1K, 6
            )
        }
 
    merged_fields = {}
    total_input_tokens = input_tokens_c
    total_output_tokens = output_tokens_c
 
    for i, page in enumerate(doc):
        text = page.get_text().strip()
 
        if text:
            messages = get_enhanced_extraction_prompt(text, i)
        else:
            base64_image = page_to_base64_image(doc, i)
            messages = get_enhanced_image_extraction_prompt(i)
            # Add the image to the user message
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
 
        input_tokens = count_tokens(messages, MODEL)
        total_input_tokens += input_tokens
 
        payload = {"messages": messages}
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
 
        if response.status_code != 200:
            return {"error": f"Extraction failed on page {i + 1}", "details": response.text}
       
       
        reply = response.json()["choices"][0]["message"]["content"]
        output_tokens = count_tokens([response.json()["choices"][0]["message"]], MODEL)
        total_output_tokens += output_tokens
 
        try:
            # Enhanced JSON parsing
            cleaned_reply = reply.strip()
            if cleaned_reply.startswith("```json"):
                cleaned_reply = cleaned_reply.removeprefix("```json").removesuffix("```").strip()
            elif cleaned_reply.startswith("```"):
                cleaned_reply = cleaned_reply.removeprefix("```").removesuffix("```").strip()
           
            if cleaned_reply.startswith("JSON:"):
                cleaned_reply = cleaned_reply.removeprefix("JSON:").strip()
               
            fields = json.loads(cleaned_reply)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{.*\}', reply, re.DOTALL)
                if json_match:
                    fields = json.loads(json_match.group())
                else:
                    fields = {}
            except:
                fields = {}
 
        if isinstance(fields, dict):
            merged_fields.update(fields)
 
    return {
        "form3200_found": True,
        "extracted_data": merged_fields,
        "input_tokens_used": total_input_tokens,
        "output_tokens_used": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": round(
            (total_input_tokens / 1000) * INPUT_COST_PER_1K +
            (total_output_tokens / 1000) * OUTPUT_COST_PER_1K, 6
        )
    }