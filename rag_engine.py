#!/usr/bin/env python3
"""
SEC-RAG engine: EDGAR download, extraction, embedding cache, and Q&A. Enhanced extraction with XBRL, structured tables, and mandatory caching
"""

from __future__ import annotations
from pathlib import Path
import requests
import datetime
import sys
import csv
import os
import io
import re
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.schema import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
import tiktoken
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
from itertools import islice
import pickle

# Optional: PDF and OCR support
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: pypdf not installed. PDF exhibit extraction will be disabled.")
        print("Install with: pip install pypdf")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract/PIL not installed. OCR will be disabled.")
    print("Install with: pip install pytesseract pillow")
    print("Also install tesseract-ocr system package")

# Configuration is read from environment variables (optionally via a .env file)
load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

tiktoken.get_encoding("cl100k_base")

# Constants
# SEC EDGAR requires a descriptive User-Agent with contact info: set SEC_USER_AGENT in .env
USER_AGENT = os.getenv("SEC_USER_AGENT", "sec-rag-demo (your-email@example.com)")
DATA_DIR = Path(__file__).with_name("data")
INDEX_DIR = Path(__file__).with_name("index")

# Optional outbound proxy, picked up from the standard environment variables
_proxy = (os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
          or os.getenv("HTTP_PROXY") or os.getenv("http_proxy"))
PROXIES = {"http": _proxy, "https": _proxy} if _proxy else None

if _proxy:
    try:
        nltk.set_proxy(_proxy)
    except Exception as e:
        print(f"Warning: Could not set NLTK proxy: {e}")


class OpenAICompatibleEmbeddings(Embeddings):
    """Embeddings client for any OpenAI-compatible /embeddings endpoint, with batch and parallel processing"""

    def __init__(self, api_key, model=DEFAULT_EMBEDDING_MODEL, base_url=DEFAULT_BASE_URL, batch_size=25, max_workers=5):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.url = base_url.rstrip("/") + "/embeddings"
        self.batch_size = batch_size  # Number of texts to send per API call
        self.max_workers = max_workers  # Number of parallel API calls

    def embed_documents(self, texts):
        """Embed a list of documents using batch + parallel processing"""
        import time
        start_time = time.time()

        # Split texts into batches
        def batched(iterable, n):
            iterator = iter(iterable)
            while batch := list(islice(iterator, n)):
                yield batch

        batches = list(batched(texts, self.batch_size))
        total_texts = len(texts)

        print(f"  → Embedding {total_texts} chunks in {len(batches)} batches (batch_size={self.batch_size}, workers={self.max_workers})")

        all_embeddings = []

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {executor.submit(self._embed_batch, batch): i for i, batch in enumerate(batches)}

            # Collect results in order
            for i in range(len(batches)):
                for future, batch_idx in future_to_batch.items():
                    if batch_idx == i:
                        batch_embeddings = future.result()
                        all_embeddings.extend(batch_embeddings)
                        break

        elapsed = time.time() - start_time
        rate = len(all_embeddings) / elapsed if elapsed > 0 else 0
        print(f"  ✅ Generated {len(all_embeddings)} embeddings in {elapsed:.1f}s ({rate:.1f} chunks/sec)")
        return all_embeddings

    def _embed_batch(self, texts):
        """Embed a batch of texts in a single API call"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Try batch API format (array of inputs)
        payload = {
            "model": self.model,
            "input": texts  # Send array of texts
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()["data"]
            # Extract embeddings in order
            return [item["embedding"] for item in sorted(data, key=lambda x: x.get("index", 0))]
        else:
            # Fallback: if batch API fails, try one at a time for this batch
            print(f"    Warning: Batch API failed ({response.status_code}), falling back to sequential")
            return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        """Embed a single query"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "input": text
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


def embed_chunk(chunk, api_key, model, base_url=DEFAULT_BASE_URL):
    """Embed a single chunk (used for parallel processing)"""
    embeddings = OpenAICompatibleEmbeddings(api_key=api_key, model=model, base_url=base_url)
    return embeddings.embed_query(chunk)


def generate_index_key(ticker, form_types, start_date, end_date):
    """Generate a unique identifier for a set of filings"""
    key_string = f"{ticker}_{'-'.join(sorted(form_types))}_{start_date}_{end_date}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_index_path(index_key):
    """Get the path to a saved FAISS index"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{index_key}.faiss"


def get_metadata_path(index_key):
    """Get the path to index metadata"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{index_key}_metadata.pkl"


def index_exists(index_key):
    """Check if a FAISS index already exists for this filing set"""
    index_path = get_index_path(index_key)
    metadata_path = get_metadata_path(index_key)
    return index_path.exists() and metadata_path.exists()


def save_index(vector_store, index_key, metadata):
    """Save a FAISS index and its metadata to disk (ALWAYS)"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = get_index_path(index_key)
    metadata_path = get_metadata_path(index_key)

    # Save FAISS index
    vector_store.save_local(str(index_path.parent), index_name=index_key)

    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✅ Index cached to disk: {index_path}")


def load_index(index_key, embeddings):
    """Load a FAISS index and its metadata from disk"""
    index_path = get_index_path(index_key)
    metadata_path = get_metadata_path(index_key)

    if not index_exists(index_key):
        return None, None

    # Load FAISS index
    vector_store = FAISS.load_local(
        str(index_path.parent),
        embeddings,
        index_name=index_key,
        allow_dangerous_deserialization=True
    )

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    print(f"✅ Index loaded from cache: {index_path}")
    return vector_store, metadata


def create_output_directory():
    """Create output directory for CSV exports"""
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def write_qa_to_csv(qa_data):
    """Write Q&A data to CSV file"""
    output_dir = create_output_directory()
    filename = f"SEC_Filings_QA_Session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Company", "Year/Quarter", "Question", "Answer"])
        writer.writerows(qa_data)

    print(f"Q&A saved to {filepath}")


def get_api_key():
    """Prompt user for API key"""
    while True:
        api_key = input("Please enter your LLM API key: ").strip()
        if api_key:
            return api_key
        print("API key cannot be empty. Please try again.")


def setup_ai():
    """Initialize AI components (LLM and embeddings)"""
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY (or OPENAI_API_KEY) not found. Set it in your environment or a .env file.")

    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        base_url=base_url,
        openai_api_key=api_key
    )
    embeddings = OpenAICompatibleEmbeddings(
        api_key=api_key,
        model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        base_url=base_url
    )
    return llm, embeddings


# ============================================================================
# ENHANCED EXTRACTION FUNCTIONS
# ============================================================================

def extract_filing_metadata(html_content):
    """Extract DEI (Document and Entity Information) metadata from filing"""
    soup = BeautifulSoup(html_content, 'html.parser')

    metadata = {}

    # Common DEI fields (these appear in inline XBRL)
    dei_fields = {
        'EntityRegistrantName': 'Company Name',
        'EntityCentralIndexKey': 'CIK',
        'DocumentType': 'Document Type',
        'DocumentPeriodEndDate': 'Period End Date',
        'CurrentFiscalYearEndDate': 'Fiscal Year End',
        'EntityCommonStockSharesOutstanding': 'Shares Outstanding',
        'DocumentFiscalYearFocus': 'Fiscal Year',
        'DocumentFiscalPeriodFocus': 'Fiscal Period',
        'EntityFilerCategory': 'Filer Category',
        'TradingSymbol': 'Trading Symbol'
    }

    for field, label in dei_fields.items():
        # Try multiple tag patterns
        for pattern in [f'dei:{field}', field]:
            tag = soup.find(attrs={'name': re.compile(pattern, re.IGNORECASE)})
            if tag:
                metadata[label] = tag.get_text(strip=True)
                break

    return metadata


def extract_xbrl_facts(html_content):
    """Extract structured XBRL/iXBRL data from filing"""
    soup = BeautifulSoup(html_content, 'html.parser')

    facts = {}

    # Find all elements with XBRL name attributes
    xbrl_patterns = [
        re.compile(r'us-gaap:', re.IGNORECASE),
        re.compile(r'dei:', re.IGNORECASE),
        re.compile(r'ifrs-full:', re.IGNORECASE)
    ]

    for pattern in xbrl_patterns:
        xbrl_tags = soup.find_all(attrs={'name': pattern})

        for tag in xbrl_tags:
            concept = tag.get('name', '')
            value = tag.get_text(strip=True)
            context = tag.get('contextref', '')
            unit = tag.get('unitref', '')
            decimals = tag.get('decimals', '')

            # Skip if value is empty or too long (likely not a fact)
            if not value or len(value) > 100:
                continue

            # Try to extract period from context
            period = None
            if context:
                context_elem = soup.find(attrs={'id': context})
                if context_elem:
                    # Look for instant or period dates
                    instant = context_elem.find(string=re.compile(r'\d{4}-\d{2}-\d{2}'))
                    if instant:
                        period = instant.strip()

            # Store the fact
            clean_concept = concept.split(':')[-1] if ':' in concept else concept

            if clean_concept not in facts:
                facts[clean_concept] = []

            facts[clean_concept].append({
                'value': value,
                'period': period,
                'unit': unit,
                'decimals': decimals,
                'context': context
            })

    return facts


def format_xbrl_facts(facts):
    """Format XBRL facts into readable text"""
    if not facts:
        return ""

    lines = ["STRUCTURED FINANCIAL DATA (XBRL):"]
    lines.append("=" * 80)

    # Key concepts to prioritize
    priority_concepts = [
        'Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent',
        'StockholdersEquity', 'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
        'NetIncomeLoss', 'OperatingIncomeLoss', 'GrossProfit',
        'EarningsPerShareBasic', 'EarningsPerShareDiluted',
        'CashAndCashEquivalentsAtCarryingValue', 'PropertyPlantAndEquipmentNet'
    ]

    # First, print priority concepts
    for concept in priority_concepts:
        if concept in facts:
            for fact in facts[concept]:
                period_str = f" ({fact['period']})" if fact['period'] else ""
                unit_str = f" {fact['unit']}" if fact['unit'] else ""
                lines.append(f"{concept}: {fact['value']}{unit_str}{period_str}")

    # Then print other significant concepts (limit to avoid clutter)
    other_count = 0
    for concept, fact_list in facts.items():
        if concept not in priority_concepts and other_count < 20:
            for fact in fact_list[:1]:  # Only first instance
                period_str = f" ({fact['period']})" if fact['period'] else ""
                lines.append(f"{concept}: {fact['value']}{period_str}")
                other_count += 1

    lines.append("=" * 80)
    return '\n'.join(lines)


def extract_tables_structured(html_content):
    """Extract tables with full structure preservation using pandas"""
    try:
        # Use pandas to read all tables
        tables = pd.read_html(io.StringIO(html_content), flavor='html5lib')

        if not tables:
            return ""

        table_texts = []
        for i, df in enumerate(tables):
            # Skip very small tables (likely navigation/formatting)
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue

            # Skip very large tables (likely data dumps)
            if df.shape[0] > 100 or df.shape[1] > 20:
                continue

            # Convert to markdown format
            try:
                table_md = df.to_markdown(index=False)
                table_texts.append(f"\n[TABLE {i+1} - {df.shape[0]} rows x {df.shape[1]} cols]\n{table_md}\n")
            except:
                # Fallback to string representation
                table_texts.append(f"\n[TABLE {i+1}]\n{df.to_string(index=False)}\n")

        return '\n'.join(table_texts)

    except Exception as e:
        print(f"Warning: Could not extract tables with pandas: {e}")
        return ""


def extract_financial_statements(html_content):
    """Identify and extract key financial statements"""
    soup = BeautifulSoup(html_content, 'html.parser')

    statements = {}

    # Common financial statement patterns
    statement_patterns = {
        'Balance Sheet': [
            'consolidated balance sheet', 'balance sheet', 'statement of financial position',
            'statement of financial condition', 'consolidated statements of financial position'
        ],
        'Income Statement': [
            'consolidated statement of income', 'income statement', 'statement of operations',
            'statement of earnings', 'consolidated statements of operations',
            'consolidated statements of income'
        ],
        'Cash Flow Statement': [
            'consolidated statement of cash flow', 'cash flow', 'statement of cash flows',
            'consolidated statements of cash flows'
        ],
        'Stockholders Equity': [
            'stockholders equity', "shareholders' equity", 'statement of equity',
            'consolidated statements of stockholders equity',
            'consolidated statements of shareholders equity'
        ]
    }

    for stmt_name, patterns in statement_patterns.items():
        for pattern in patterns:
            # Find text elements that might be headers
            candidates = soup.find_all(['p', 'div', 'span', 'b', 'strong', 'h1', 'h2', 'h3', 'h4'],
                                      string=re.compile(pattern, re.IGNORECASE))

            for candidate in candidates:
                # Look for the next table after this header
                table = candidate.find_next('table')

                if table:
                    try:
                        # Try to parse with pandas
                        df_list = pd.read_html(str(table))
                        if df_list:
                            df = df_list[0]

                            # Only accept if it looks like a financial statement
                            if df.shape[0] >= 3 and df.shape[1] >= 2:
                                statements[stmt_name] = df.to_markdown(index=False)
                                print(f"  → Found: {stmt_name}")
                                break  # Found this statement, move to next
                    except Exception as e:
                        continue

            if stmt_name in statements:
                break  # Found this statement type, stop searching patterns

    return statements


def format_financial_statements(statements):
    """Format extracted financial statements into text"""
    if not statements:
        return ""

    lines = ["FINANCIAL STATEMENTS:"]
    lines.append("=" * 80)

    for stmt_name, stmt_content in statements.items():
        lines.append(f"\n{stmt_name.upper()}:")
        lines.append("-" * 80)
        lines.append(stmt_content)
        lines.append("")

    lines.append("=" * 80)
    return '\n'.join(lines)


# ============================================================================
# EXHIBIT EXTRACTION (PDF and OCR)
# ============================================================================

def find_exhibit_links(soup, base_url):
    """
    Find links to exhibits (PDFs, images) in the filing HTML.

    Args:
        soup: BeautifulSoup object of the filing HTML
        base_url: Base URL for constructing full URLs

    Returns:
        list: List of exhibit dictionaries with URL, type, and metadata
    """
    exhibits = []

    # Find image tags (charts, graphs, infographics)
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if not src:
            continue

        # Construct full URL
        if src.startswith('http'):
            full_url = src
        elif src.startswith('/'):
            # Absolute path from SEC domain
            full_url = f"https://www.sec.gov{src}"
        else:
            # Relative path
            full_url = f"{base_url}/{src}"

        # Get context (surrounding text)
        parent = img.find_parent()
        context = parent.get_text(strip=True)[:200] if parent else ""

        exhibits.append({
            'type': 'image',
            'url': full_url,
            'alt': img.get('alt', ''),
            'context': context,
            'filename': src.split('/')[-1]
        })

    # Find links to PDFs and other exhibits
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)

        # Skip if no text or not an exhibit
        if not text:
            continue

        # Look for exhibit indicators
        text_lower = text.lower()
        is_exhibit = any(x in text_lower for x in ['exhibit', 'ex ', 'ex-', 'ex.', 'graphic'])

        # Or look for file extensions
        is_pdf = href.lower().endswith('.pdf')
        is_image = any(href.lower().endswith(ext) for ext in ['.png', '.jpg', '.gif', '.jpeg', '.bmp'])

        if is_exhibit or is_pdf or is_image:
            # Construct full URL
            if href.startswith('http'):
                full_url = href
            elif href.startswith('/'):
                full_url = f"https://www.sec.gov{href}"
            else:
                full_url = f"{base_url}/{href}"

            exhibits.append({
                'type': 'pdf' if is_pdf else ('image' if is_image else 'exhibit'),
                'url': full_url,
                'name': text,
                'filename': href.split('/')[-1]
            })

    return exhibits


def extract_pdf_exhibit(pdf_url):
    """
    Download and extract text from a PDF exhibit.

    Args:
        pdf_url: URL to the PDF file

    Returns:
        str: Extracted text or error message
    """
    if not PDF_AVAILABLE:
        return "[PDF extraction not available - pypdf not installed]"

    try:
        # Download PDF
        response = requests.get(
            pdf_url,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30,
            stream=True
        )
        response.raise_for_status()

        # Check file size (skip if > 10MB)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > 10 * 1024 * 1024:
            return f"[PDF too large: {int(content_length) / (1024*1024):.1f}MB - skipped]"

        # Read PDF
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)

        # Extract text from all pages
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[Page {i+1}]\n{page_text}")
            except Exception as e:
                text_parts.append(f"[Page {i+1} - Error: {str(e)}]")

        if not text_parts:
            return "[PDF contains no extractable text - may be scanned images]"

        return '\n\n'.join(text_parts)

    except Exception as e:
        return f"[Error extracting PDF: {str(e)}]"


def extract_text_from_image_ocr(image_url):
    """
    Download an image and extract text using OCR.

    Args:
        image_url: URL to the image file

    Returns:
        str: Extracted text or error message
    """
    if not OCR_AVAILABLE:
        return "[OCR not available - pytesseract/PIL not installed]"

    try:
        # Download image
        response = requests.get(
            image_url,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        response.raise_for_status()

        # Check file size (skip if > 5MB)
        if len(response.content) > 5 * 1024 * 1024:
            return f"[Image too large: {len(response.content) / (1024*1024):.1f}MB - skipped]"

        # Open image
        img = Image.open(io.BytesIO(response.content))

        # Convert to RGB if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Extract text using OCR
        text = pytesseract.image_to_string(img)

        if not text.strip():
            return "[No text found in image]"

        return text.strip()

    except Exception as e:
        return f"[Error processing image: {str(e)}]"


def extract_exhibits(html_content, base_url, max_exhibits=10):
    """
    Extract text from PDF and image exhibits in the filing.

    Args:
        html_content: HTML content of the filing
        base_url: Base URL for the filing (for resolving relative links)
        max_exhibits: Maximum number of exhibits to process (default: 10)

    Returns:
        str: Formatted text with exhibit content
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    print("  → Finding exhibit links...")
    exhibits = find_exhibit_links(soup, base_url)

    if not exhibits:
        print("     No exhibits found")
        return ""

    print(f"     Found {len(exhibits)} exhibit links")

    # Prioritize PDFs over images
    pdfs = [ex for ex in exhibits if ex['type'] == 'pdf']
    images = [ex for ex in exhibits if ex['type'] == 'image']

    # Limit processing
    pdfs = pdfs[:max_exhibits // 2]  # Process half the limit as PDFs
    images = images[:max_exhibits // 2]  # Process half as images

    exhibit_texts = []

    # Process PDFs
    if pdfs and PDF_AVAILABLE:
        print(f"  → Extracting {len(pdfs)} PDF exhibits...")
        for i, exhibit in enumerate(pdfs, 1):
            print(f"     [{i}/{len(pdfs)}] {exhibit['filename']}")
            text = extract_pdf_exhibit(exhibit['url'])

            if text and not text.startswith('['):  # Skip error messages
                exhibit_texts.append(f"""
{'=' * 80}
EXHIBIT: {exhibit['name']}
TYPE: PDF
FILE: {exhibit['filename']}
URL: {exhibit['url']}
{'=' * 80}

{text}
""")

    # Process images with OCR
    if images and OCR_AVAILABLE:
        print(f"  → Extracting text from {len(images)} images using OCR...")
        for i, exhibit in enumerate(images, 1):
            print(f"     [{i}/{len(images)}] {exhibit['filename']}")
            text = extract_text_from_image_ocr(exhibit['url'])

            if text and not text.startswith('['):  # Skip error messages
                exhibit_texts.append(f"""
{'=' * 80}
EXHIBIT: {exhibit.get('alt', 'Image')}
TYPE: Image (OCR)
FILE: {exhibit['filename']}
CONTEXT: {exhibit['context']}
URL: {exhibit['url']}
{'=' * 80}

{text}
""")

    if not exhibit_texts:
        return ""

    print(f"  ✅ Successfully extracted {len(exhibit_texts)} exhibits")

    return '\n\n'.join(exhibit_texts)


def extract_text_with_page_tracking(html_content, quick_mode=False):
    """
    Extract text from SEC HTML filing with actual page numbers tracked.

    SEC HTML filings use <hr style="page-break-after:always"> tags to mark page boundaries.
    This function splits the document by these markers and tracks the actual page number
    for each segment.

    Args:
        html_content: The HTML content to extract from
        quick_mode: If True, skip slow XBRL/table extraction

    Returns:
        List of (page_text, page_number, page_metadata) tuples where:
        - page_text: Extracted text for this page (with XBRL/tables if not quick_mode)
        - page_number: Actual page number (1-indexed)
        - page_metadata: Dict with metadata extracted from this page
    """
    import re

    print("  → Extracting text with page number tracking...")

    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Convert to string for splitting by page breaks
    html_str = str(soup)

    # Split by page break markers
    page_break_pattern = r'<hr[^>]*page-break[^>]*/?>'
    pages_html = re.split(page_break_pattern, html_str)

    print(f"  → Found {len(pages_html)} pages in document")

    # Extract XBRL and metadata once if not in quick mode
    # (these apply to the whole document, not individual pages)
    if not quick_mode:
        print("  → Extracting document-level metadata and XBRL facts...")
        metadata = extract_filing_metadata(html_content)
        xbrl_facts = extract_xbrl_facts(html_content)
        xbrl_text = format_xbrl_facts(xbrl_facts)
    else:
        metadata = None
        xbrl_text = None

    # Process each page
    page_segments = []
    for page_num, page_html in enumerate(pages_html, start=1):
        # Parse page HTML
        page_soup = BeautifulSoup(page_html, 'html.parser')

        # Remove script and style elements
        for script in page_soup(['script', 'style']):
            script.extract()

        # Extract text
        page_text = page_soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in page_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Only include non-empty pages
        if cleaned_text.strip():
            # Add document-level XBRL to first page only (to avoid duplication)
            if page_num == 1 and xbrl_text:
                cleaned_text = xbrl_text + "\n\n" + cleaned_text

            page_metadata = {
                'page_number': page_num,
                'page_length': len(cleaned_text)
            }

            # Add document metadata to first page
            if page_num == 1 and metadata:
                page_metadata['filing_metadata'] = metadata

            page_segments.append((cleaned_text, page_num, page_metadata))

    print(f"  ✅ Extracted {len(page_segments)} non-empty pages with actual page numbers")

    return page_segments


def extract_text_from_html_enhanced(html_content, base_url=None, include_exhibits=True, quick_mode=False, track_pages=False):
    """
    Enhanced text extraction with XBRL, structured tables, and metadata.
    This is the main extraction function that combines all enhancements.

    Args:
        html_content: The HTML content to extract from
        base_url: Base URL for resolving relative URLs (for exhibits)
        include_exhibits: Whether to extract PDF/image exhibits (slow)
        quick_mode: If True, skip slow XBRL/table extraction (for batch analysis)
        track_pages: If True, return list of (text, page_num, metadata) tuples instead of combined text

    Returns:
        If track_pages=False: String with all extracted text combined
        If track_pages=True: List of (page_text, page_number, page_metadata) tuples
    """
    # If page tracking requested, use specialized function
    if track_pages:
        return extract_text_with_page_tracking(html_content, quick_mode=quick_mode)

    # Otherwise, use original combined extraction
    soup = BeautifulSoup(html_content, 'html.parser')

    if quick_mode:
        # Fast mode - skip slow extractions
        print("  → Fast mode: skipping XBRL/tables/statements (extracting body text only)")
        metadata = None
        xbrl_facts = {}
        financial_statements = {}
        tables_text = None
    else:
        # Full mode - extract everything
        print("  → Extracting filing metadata...")
        metadata = extract_filing_metadata(html_content)

        print("  → Extracting XBRL facts...")
        xbrl_facts = extract_xbrl_facts(html_content)

        print("  → Identifying financial statements...")
        financial_statements = extract_financial_statements(html_content)

        print("  → Extracting structured tables...")
        tables_text = extract_tables_structured(html_content)

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get main text content
    print("  → Extracting body text...")
    text = soup.get_text(separator="\n")

    # Clean up text - remove extra whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)

    # Combine all components with clear section markers
    components = []

    # Add metadata section
    if metadata:
        meta_lines = ["FILING METADATA:", "=" * 80]
        for key, value in metadata.items():
            meta_lines.append(f"{key}: {value}")
        meta_lines.append("=" * 80)
        components.append('\n'.join(meta_lines))

    # Add XBRL facts
    xbrl_text = format_xbrl_facts(xbrl_facts)
    if xbrl_text:
        components.append(xbrl_text)

    # Add financial statements
    statements_text = format_financial_statements(financial_statements)
    if statements_text:
        components.append(statements_text)

    # Add structured tables
    if tables_text:
        components.append(f"\nADDITIONAL TABLES:\n{tables_text}")

    # Add exhibits (PDFs and OCR images)
    if include_exhibits and base_url:
        exhibits_text = extract_exhibits(html_content, base_url, max_exhibits=10)
        if exhibits_text:
            components.append(f"\nEXHIBITS:\n{exhibits_text}")

    # Add main text content
    components.append(f"\nFULL DOCUMENT TEXT:\n{'=' * 80}\n{cleaned_text}")

    # Combine everything
    full_text = '\n\n'.join(components)

    print(f"  ✅ Extracted {len(full_text):,} characters")

    return full_text


def extract_text_from_html(html_content):
    """
    Backward compatibility wrapper for extract_text_from_html_enhanced.
    Used by app.py.

    Note: This simplified version doesn't include exhibits.
    For full functionality, use extract_text_from_html_enhanced directly.
    """
    # Call enhanced version without exhibits for compatibility
    return extract_text_from_html_enhanced(html_content, base_url=None, include_exhibits=False)


# ============================================================================
# FILING DOWNLOAD AND PROCESSING
# ============================================================================

def get_available_form_types(ticker):
    """
    Query SEC to get the actual available form types for a specific company.

    Args:
        ticker: Company ticker symbol

    Returns:
        tuple: (list of form types with counts, company_name, date_range) or (None, None, None) on error
    """
    try:
        # Get CIK from ticker
        tickerdict = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            proxies=PROXIES,
            timeout=10,
        ).json()

        cik_str = None
        company_name = None
        for entry in tickerdict.values():
            if entry["ticker"].upper() == ticker.upper():
                cik_str = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break

        if not cik_str:
            return None, None, None

        # Get submissions metadata
        SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        submissions_response = requests.get(
            SUBMISSIONS_URL,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()

        # Extract form types and count occurrences
        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        filing_dates = recent["filingDate"]

        # Count each form type
        form_counts = {}
        for form in forms:
            clean_form = (form or "").strip().upper()
            if clean_form:
                form_counts[clean_form] = form_counts.get(clean_form, 0) + 1

        # Sort by count (most common first)
        sorted_forms = sorted(form_counts.items(), key=lambda x: x[1], reverse=True)

        # Get date range
        all_dates = [d for d in filing_dates if d]
        date_range = (min(all_dates), max(all_dates)) if all_dates else (None, None)

        return sorted_forms, company_name, date_range

    except Exception as e:
        print(f"Warning: Could not fetch form types: {e}")
        return None, None, None


def get_company_filings(ticker, requested_forms, start_date, end_date, interactive=True, selected_indices=None):
    """
    Download filings for a specific ticker (streaming, in-memory processing).
    Returns text segments ready for embedding without saving HTML to disk.

    Args:
        ticker: Company ticker symbol
        requested_forms: List of form types to download
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interactive: If True, show available filings and let user select (default: True)
        selected_indices: Pre-selected filing indices (for batch/GUI mode)
    """
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {ticker}")
    print(f"{'=' * 80}")
    print(f"Looking up CIK for ticker: {ticker}")

    try:
        # Step 1: Get CIK from ticker
        tickerdict = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            proxies=PROXIES,
            timeout=10,
        ).json()

        cik_str = None
        company_name = None
        for entry in tickerdict.values():
            if entry["ticker"].upper() == ticker.upper():
                cik_str = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                print(f"✅ Found CIK: {cik_str} for {company_name}")
                break

        if not cik_str:
            print(f"❌ Error: Ticker {ticker} not found in SEC database.")
            return None, None, None

        # Step 2: Get submissions metadata
        SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        print(f"Fetching filing list from SEC...")

        submissions_response = requests.get(
            SUBMISSIONS_URL,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()

        # Step 3: Extract and filter filings
        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        accessions = recent["accessionNumber"]
        primary_docs = recent["primaryDocument"]
        filing_dates = recent["filingDate"]

        # Get overall date range for all filings
        all_dates = [d for d in filing_dates if d]
        if all_dates:
            earliest_date = min(all_dates)
            latest_date = max(all_dates)
            print(f"\n📅 Available filing date range for {company_name}:")
            print(f"   Earliest: {earliest_date}")
            print(f"   Latest: {latest_date}")
            print(f"   Requested: {start_date} to {end_date}")

        # Filter by form type and date range
        filtered_indices = [
            idx for idx, (form, date) in enumerate(zip(forms, filing_dates))
            if ((form or "").strip().upper() in requested_forms and
                start_date <= date <= end_date)
        ]

        if not filtered_indices:
            print(f"\n❌ No matching filings found for the selected criteria.")
            print(f"   Requested forms: {', '.join(requested_forms)}")
            print(f"   Date range: {start_date} to {end_date}")
            return None, None, None

        print(f"\n✅ Found {len(filtered_indices)} matching filings:")
        print("=" * 80)

        # Display available filings
        for i, idx in enumerate(filtered_indices):
            print(f"{i+1}. {forms[idx]} filed on {filing_dates[idx]} (Accession: {accessions[idx]})")

        print("=" * 80)

        # Step 4: Get user selection (if interactive)
        if interactive and selected_indices is None:
            print("\nOptions:")
            print("  - Press ENTER to download ALL filings")
            print("  - Enter numbers (comma-separated, e.g., 1,3,5) to select specific filings")
            print("  - Enter 'cancel' to skip this company")

            selection = input("\nYour choice: ").strip().lower()

            if selection == 'cancel':
                print("Cancelled.")
                return None, None, None
            elif selection == '':
                # Download all
                selected_indices = filtered_indices
                print(f"\n✅ Will download all {len(selected_indices)} filings")
            else:
                # Parse user selection
                try:
                    selected_numbers = [int(x.strip()) for x in selection.split(',')]
                    selected_indices = [filtered_indices[n-1] for n in selected_numbers if 0 < n <= len(filtered_indices)]
                    print(f"\n✅ Will download {len(selected_indices)} selected filings")
                except (ValueError, IndexError) as e:
                    print(f"❌ Invalid selection: {e}")
                    return None, None, None
        elif selected_indices is None:
            # Non-interactive mode: download all
            selected_indices = filtered_indices

        if not selected_indices:
            print("❌ No filings selected.")
            return None, None, None

        # Step 5: Process each selected filing (streaming - no local save)
        text_segments = []

        for idx in selected_indices:
            accession = accessions[idx]
            primary_document = primary_docs[idx]
            filing_date = filing_dates[idx]
            form_type = forms[idx]

            # Construct document URL
            cik_nozero = cik_str.lstrip("0")
            accession_nodash = accession.replace("-", "")
            document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_nodash}/{primary_document}"

            print(f"\n📥 Downloading {form_type} filed on {filing_date}...")
            print(f"   URL: {document_url}")

            # Download filing HTML (keep in memory only)
            document_response = requests.get(
                document_url,
                headers={"User-Agent": USER_AGENT},
                proxies=PROXIES,
                timeout=30,
            )
            document_response.raise_for_status()

            html_content = document_response.text

            # Extract all information from this filing (including exhibits)
            # Base URL is the directory containing the document (for resolving relative links)
            base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_nodash}"

            print(f"📄 Processing {form_type} content...")

            # Check if this is just a cover page with exhibit references
            # Common for 6-K, 8-K forms where actual data is in exhibits
            if len(html_content) < 50000 and 'exhibit' in html_content.lower():
                print(f"  ⚠️  Detected cover page with exhibits - looking for financial data exhibits...")

                # Parse HTML to find exhibit links
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                financial_exhibits = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text(strip=True).lower()

                    # Look for financial statement exhibits
                    if any(keyword in text for keyword in ['financial statement', 'interim', 'earnings', 'results', 'condensed consolidated']):
                        if href.endswith('.htm') or href.endswith('.html'):
                            exhibit_url = f"{base_url}/{href}"
                            print(f"  → Found financial exhibit: {href}")
                            financial_exhibits.append((href, exhibit_url))

                # Download and include the first financial exhibit
                if financial_exhibits:
                    exhibit_name, exhibit_url = financial_exhibits[0]
                    print(f"  📥 Downloading exhibit: {exhibit_name}...")

                    try:
                        exhibit_response = requests.get(
                            exhibit_url,
                            headers={"User-Agent": USER_AGENT},
                            proxies=PROXIES,
                            timeout=30
                        )
                        exhibit_response.raise_for_status()

                        # Add exhibit content to the main HTML
                        exhibit_html = exhibit_response.text
                        html_content = html_content + "\n\n<!-- EXHIBIT CONTENT -->\n\n" + exhibit_html
                        print(f"  ✅ Added exhibit content ({len(exhibit_html):,} characters)")
                    except Exception as e:
                        print(f"  ⚠️  Failed to download exhibit: {e}")

            # Extract with page tracking for accurate page numbers
            page_segments = extract_text_from_html_enhanced(
                html_content,
                base_url=base_url,
                include_exhibits=True,
                track_pages=True  # Enable page tracking
            )

            # page_segments is a list of (page_text, page_number, page_metadata) tuples
            # Transform into format expected by prepare_document_with_metadata

            # Fallback: If extraction fails (empty or very short), use simple extraction
            if not page_segments or (len(page_segments) == 1 and len(page_segments[0][0]) < 5000):
                print(f"  ⚠️  Enhanced extraction returned insufficient data - using simple extraction fallback")

                # Use simple extraction
                soup = BeautifulSoup(html_content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                # Get text
                text = soup.get_text(separator="\n")

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                extracted_text = '\n'.join(chunk for chunk in chunks if chunk)

                print(f"  ✅ Simple extraction: {len(extracted_text):,} characters")

                # Create single page segment with no page tracking
                page_segments = [(extracted_text, None, {})]

            # Add filing header to first page
            filing_header = f"""
{'=' * 80}
FILING TYPE: {form_type}
FILING DATE: {filing_date}
ACCESSION NUMBER: {accession}
DOCUMENT: {primary_document}
SOURCE URL: {document_url}
{'=' * 80}
"""

            # Process each page segment
            for page_idx, (page_text, page_number, page_meta) in enumerate(page_segments):
                # Add header only to first page
                if page_idx == 0:
                    full_text = filing_header + "\n\n" + page_text
                else:
                    full_text = page_text

                # Build metadata for this page
                metadata = {
                    'form_type': form_type,
                    'filing_date': filing_date,
                    'accession': accession,
                    'ticker': ticker,
                    'company_name': company_name,
                }

                # Add page number if available (actual page, not estimated)
                if page_number is not None:
                    metadata['page_number'] = page_number

                # Store text segment with metadata
                text_segments.append((full_text, metadata))

            print(f"✅ Completed processing {form_type}")

            # html_content goes out of scope here and is garbage collected

        # Generate filing info summary
        filing_info = f"{', '.join(requested_forms)} ({start_date} to {end_date})"

        print(f"\n{'=' * 80}")
        print(f"✅ DOWNLOAD COMPLETE: Processed {len(text_segments)} filings")
        print(f"{'=' * 80}\n")

        return text_segments, company_name, filing_info

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def prepare_document(text_segments, embeddings, index_key, force_rebuild=False):
    """
    Prepare document for analysis with MANDATORY caching.

    Args:
        text_segments: List of (text, metadata) tuples
        embeddings: The embeddings model to use
        index_key: Cache key (REQUIRED)
        force_rebuild: If True, rebuilds index even if cache exists

    Returns:
        vector_store: FAISS vector store ready for queries
    """

    # Check if we can load from cache
    if not force_rebuild and index_exists(index_key):
        print(f"\n⚡ Loading pre-computed index from cache...")
        vector_store, metadata = load_index(index_key, embeddings)
        if vector_store:
            print(f"✅ Loaded {metadata.get('num_chunks', 'unknown')} chunks from cache")
            return vector_store

    print(f"\n{'=' * 80}")
    print("BUILDING VECTOR STORE")
    print(f"{'=' * 80}")

    # Prepare documents with metadata
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=400
    )

    all_docs = []
    for text, metadata in text_segments:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = LangChainDocument(page_content=chunk, metadata=metadata)
            all_docs.append(doc)

    print(f"✅ Created {len(all_docs)} chunks from {len(text_segments)} filings")

    # Generate embeddings in parallel
    print(f"\n🔄 Generating embeddings using {multiprocessing.cpu_count()} CPU cores...")
    print("   This may take several minutes for large filings...")

    chunks_list = [doc.page_content for doc in all_docs]
    embeddings_list = []

    num_processes = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_chunk = {
            executor.submit(embed_chunk, chunk, embeddings.api_key, embeddings.model, embeddings.base_url): chunk
            for chunk in chunks_list
        }

        completed = 0
        for future in as_completed(future_to_chunk):
            try:
                embedding = future.result()
                embeddings_list.append(embedding)
                completed += 1
                if completed % 10 == 0 or completed == len(chunks_list):
                    print(f"   Progress: {completed}/{len(chunks_list)} chunks embedded ({completed/len(chunks_list)*100:.1f}%)")
            except Exception as exc:
                print(f'   ⚠️ Chunk embedding failed: {exc}')

    # Create FAISS vector store
    print("\n🔨 Building FAISS index...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(chunks_list, embeddings_list)),
        embedding=embeddings
    )

    print("✅ Vector store created")

    # ALWAYS save to cache
    metadata = {
        'num_chunks': len(all_docs),
        'num_filings': len(text_segments),
        'chunk_size': 8000,
        'chunk_overlap': 400,
        'created_at': datetime.datetime.now().isoformat(),
        'index_key': index_key
    }

    print(f"\n💾 Saving to cache (mandatory)...")
    save_index(vector_store, index_key, metadata)

    print(f"{'=' * 80}")
    print("✅ VECTOR STORE READY")
    print(f"{'=' * 80}\n")

    return vector_store


def analyze_text(vector_store, llm, user_question, num_chunks=10):
    """Query the vector store and get an answer from the LLM"""
    print(f"\n🔍 Searching for top {num_chunks} most relevant chunks...")
    results = vector_store.similarity_search(user_question, k=num_chunks)

    result_docs = "\n\n".join([
        f'Chunk {i+1} [from {doc.metadata.get("form_type", "Unknown")} filed {doc.metadata.get("filing_date", "Unknown")}]:\n{doc.page_content}\n'
        for i, doc in enumerate(results)
    ])

    print(f"✅ Retrieved {len(results)} relevant chunks")

    print("\n🤖 Sending to the LLM for analysis...")
    llm_output = llm.invoke([HumanMessage(content=f"""From the perspective of an experienced financial analyst, answer the question using the context provided from the SEC filings.

Question: {user_question}

Context from SEC filings:

{result_docs}

Please provide a comprehensive answer based on the context above.""")])

    print("✅ Answer generated\n")

    return llm_output.content, result_docs


def ask_questions(vector_store, llm, company, filing_info):
    """Interactive Q&A session"""
    qa_list = []
    while True:
        user_question = input("\n" + "="*80 + "\nPlease enter your question about the filing (or type 'exit' to end): ")
        if user_question.lower() == 'exit':
            break

        answer, sources = analyze_text(vector_store, llm, user_question)

        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        print("="*80)

        qa_list.append([company, filing_info, user_question, answer])

        # Ask if the user wants to save sources
        save_sources = input("\nDo you want to save the source chunks for this question? (yes/no): ").strip().lower()
        if save_sources == 'yes':
            output_dir = create_output_directory()
            filename = f"sources_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Question: {user_question}\n\n")
                f.write(f"Answer: {answer}\n\n")
                f.write("Sources:\n")
                f.write(sources)
            print(f"✅ Sources saved to {filepath}")

    return qa_list


def ask_to_continue():
    """Ask user if they want to analyze another company"""
    while True:
        response = input("\nDo you want to analyze another company's filings? (yes/no): ").strip().lower()
        if response in ['yes', 'no']:
            return response == 'yes'
        print("Please enter 'yes' or 'no'.")


# ============================================================================
# MAIN INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive analysis mode for single company"""
    print("\n" + "="*80)
    print("SEC-RAG - INTERACTIVE MODE")
    print("Enhanced with XBRL extraction and mandatory caching")
    print("="*80)

    # Initialize AI
    llm, embeddings = setup_ai()

    while True:
        # Get ticker
        ticker = input("\nEnter a ticker symbol (e.g., AAPL, MSFT, KLAR): ").strip().upper()
        if not ticker:
            print("❌ Ticker is required.")
            continue

        # Fetch available form types for this specific company
        print(f"\nFetching available form types for {ticker}...")
        form_types, company_name_preview, date_range = get_available_form_types(ticker)

        if form_types:
            print(f"\n✅ Found {len(form_types)} form types for {company_name_preview}:")
            print("=" * 80)

            # Show top 15 most common form types
            for form_type, count in form_types[:15]:
                print(f"  {form_type:15s} ({count} filings)")

            if len(form_types) > 15:
                print(f"  ... and {len(form_types) - 15} more form types")

            print("=" * 80)

            if date_range[0] and date_range[1]:
                print(f"Filing date range: {date_range[0]} to {date_range[1]}")
        else:
            print(f"⚠️  Could not fetch form types for {ticker}.")
            print("   Common form types: 10-K, 10-Q, 8-K, F-1, 6-K, DEF 14A, S-1")

        # Get form types from user
        form_input = input("\nEnter form types (comma-separated, e.g., 10-K,10-Q or F-1,6-K): ").strip().upper()
        requested_forms = [f.strip() for f in form_input.split(',') if f.strip()]

        if not requested_forms:
            print("❌ At least one form type is required.")
            continue

        # Get date range
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()

        # Generate cache key
        index_key = generate_index_key(ticker, requested_forms, start_date, end_date)

        # Check if cached
        if index_exists(index_key):
            print(f"\n✅ Found cached embeddings for this query!")
            use_cache = input("Load from cache? (yes/no): ").strip().lower()

            if use_cache == 'yes':
                vector_store = load_index(index_key, embeddings)[0]
                company = ticker  # We don't have company name from cache
                filing_info = f"{', '.join(requested_forms)} ({start_date} to {end_date})"

                # Skip to Q&A
                qa_list = ask_questions(vector_store, llm, company, filing_info)

                if qa_list:
                    write_qa_to_csv(qa_list)

                if not ask_to_continue():
                    break
                continue

        # Download and process filings
        text_segments, company_name, filing_info = get_company_filings(
            ticker, requested_forms, start_date, end_date
        )

        if not text_segments:
            print("❌ Failed to download filings. Please try again.")
            if not ask_to_continue():
                break
            continue

        # Prepare vector store (will cache automatically)
        vector_store = prepare_document(text_segments, embeddings, index_key)

        # Q&A session
        qa_list = ask_questions(vector_store, llm, company_name, filing_info)

        # Save results
        if qa_list:
            write_qa_to_csv(qa_list)

        if not ask_to_continue():
            break

    print("\n✅ Thank you for using SEC-RAG!")


# ============================================================================
# BATCH MODE
# ============================================================================

def batch_mode():
    """Batch analysis mode for multiple companies/questions"""
    print("\n" + "="*80)
    print("SEC-RAG - BATCH MODE")
    print("="*80)

    # Initialize AI
    llm, embeddings = setup_ai()

    # Get batch parameters
    tickers_input = input("\nEnter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOG): ").strip().upper()
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    form_input = input("Enter form types (comma-separated, e.g., 10-K,10-Q): ").strip().upper()
    requested_forms = [f.strip() for f in form_input.split(',') if f.strip()]

    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    print("\nEnter questions (one per line, press Enter twice when done):")
    questions = []
    while True:
        q = input().strip()
        if not q:
            break
        questions.append(q)

    if not tickers or not requested_forms or not questions:
        print("❌ All fields are required for batch mode.")
        return

    # Process each ticker
    all_qa = []

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING TICKER: {ticker}")
        print(f"{'=' * 80}")

        # Generate cache key
        index_key = generate_index_key(ticker, requested_forms, start_date, end_date)

        # Try to load from cache
        if index_exists(index_key):
            print(f"✅ Loading from cache...")
            vector_store = load_index(index_key, embeddings)[0]
            company_name = ticker
            filing_info = f"{', '.join(requested_forms)} ({start_date} to {end_date})"
        else:
            # Download and process (non-interactive mode)
            text_segments, company_name, filing_info = get_company_filings(
                ticker, requested_forms, start_date, end_date, interactive=False
            )

            if not text_segments:
                print(f"❌ Failed to process {ticker}, skipping...")
                continue

            # Prepare vector store (will cache automatically)
            vector_store = prepare_document(text_segments, embeddings, index_key)

        # Answer each question
        for question in questions:
            print(f"\n📝 Question: {question}")
            answer, _ = analyze_text(vector_store, llm, question)
            all_qa.append([company_name, filing_info, question, answer])

    # Save all results
    if all_qa:
        write_qa_to_csv(all_qa)
        print(f"\n✅ Batch analysis complete! Generated {len(all_qa)} Q&A pairs.")
    else:
        print("\n❌ No results to save.")


# ============================================================================
# PRE-PROCESSING MODE
# ============================================================================

def preprocess_mode():
    """Pre-process filings to build cache without asking questions"""
    print("\n" + "="*80)
    print("SEC-RAG - PRE-PROCESSING MODE")
    print("Build embeddings cache for future fast queries")
    print("="*80)

    # Initialize AI
    llm, embeddings = setup_ai()

    # Get parameters
    tickers_input = input("\nEnter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOG): ").strip().upper()
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    form_input = input("Enter form types (comma-separated, e.g., 10-K,10-Q): ").strip().upper()
    requested_forms = [f.strip() for f in form_input.split(',') if f.strip()]

    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    if not tickers or not requested_forms:
        print("❌ Tickers and form types are required.")
        return

    successful = []
    failed = []
    skipped = []

    for i, ticker in enumerate(tickers):
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{len(tickers)}] PRE-PROCESSING: {ticker}")
        print(f"{'=' * 80}")

        # Generate cache key
        index_key = generate_index_key(ticker, requested_forms, start_date, end_date)

        # Check if already cached
        if index_exists(index_key):
            print(f"✅ {ticker} already cached, skipping...")
            skipped.append(ticker)
            continue

        try:
            # Download and process (non-interactive mode)
            text_segments, company_name, filing_info = get_company_filings(
                ticker, requested_forms, start_date, end_date, interactive=False
            )

            if not text_segments:
                print(f"❌ Failed to download filings for {ticker}")
                failed.append(ticker)
                continue

            # Build vector store (will cache automatically)
            vector_store = prepare_document(text_segments, embeddings, index_key)

            successful.append(ticker)
            print(f"✅ {ticker} successfully pre-processed and cached!")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            failed.append(ticker)

    # Summary
    print(f"\n{'=' * 80}")
    print("PRE-PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ Successful: {len(successful)} - {', '.join(successful) if successful else 'None'}")
    print(f"⏭️  Skipped (already cached): {len(skipped)} - {', '.join(skipped) if skipped else 'None'}")
    print(f"❌ Failed: {len(failed)} - {', '.join(failed) if failed else 'None'}")
    print(f"{'=' * 80}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with mode selection"""
    print("\n" + "="*80)
    print("SEC-RAG")
    print("Enhanced with XBRL, structured tables, and mandatory caching")
    print("="*80)
    print("\nSelect mode:")
    print("1. Interactive Mode - Analyze one company at a time")
    print("2. Batch Mode - Analyze multiple companies/questions")
    print("3. Pre-processing Mode - Build cache for future queries")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        interactive_mode()
    elif choice == '2':
        batch_mode()
    elif choice == '3':
        preprocess_mode()
    elif choice == '4':
        print("\n✅ Goodbye!")
    else:
        print("\n❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
