#!/usr/bin/env python3
"""
Gradio GUI for SEC Filing Analysis Tool
"""

import os
# Disable Gradio analytics and network checks to avoid proxy issues
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
# Keep Gradio's local traffic off any configured outbound proxy
os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "localhost", "127.0.0.1"]))

import gradio as gr
import pandas as pd
import datetime
from pathlib import Path
import pickle
import tiktoken

# Import functions from rag_engine (enhanced with XBRL, OCR, PDF extraction)
from rag_engine import (
    setup_ai,
    get_company_filings,
    extract_text_from_html,  # Backward compatible wrapper
    extract_text_from_html_enhanced,  # Enhanced version with exhibits
    prepare_document,
    analyze_text,
    generate_index_key,
    index_exists,
    load_index,
    save_index,
    get_index_path,
    get_metadata_path,
    get_available_form_types,  # NEW: Dynamic form type detection
    INDEX_DIR,
    DATA_DIR,
    USER_AGENT,
    PROXIES
)

# Import FAISS for vector store operations
from langchain_community.vectorstores import FAISS

import requests

# For document processing
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

# Global state
llm = None
embeddings = None
vector_store = None
current_company = None
current_filing_info = None
qa_history = []
document_sources = []  # Track which documents were loaded: [(doc_type, doc_name, start_char, end_char), ...]

# Cost tracking
session_input_tokens = 0
session_output_tokens = 0
session_total_cost = 0.0

def initialize_ai():
    """Initialize AI components"""
    global llm, embeddings
    if llm is None or embeddings is None:
        try:
            llm, embeddings = setup_ai()
            return True, "[SUCCESS] AI components initialized successfully"
        except Exception as e:
            return False, f"[ERROR] Failed to initialize AI: {e}\n\nPlease ensure LLM_API_KEY is set in your environment or a .env file (see .env.example)"
    return True, "[SUCCESS] AI components already initialized"

def estimate_tokens(text, model="claude"):
    """Estimate token count for text using tiktoken"""
    # Proxy for Claude (cl100k_base) vs Llama 3 (o200k_base)
    enc_name = "o200k_base" if model == "llama" else "cl100k_base"
    enc = tiktoken.get_encoding(enc_name)
    base = len(enc.encode(text))
    # Add 15% buffer for flagship Claude dictionary
    return int(base * 1.15) if model == "claude" else base

def calculate_cost(input_tokens, output_tokens):
    """Estimate cost from per-million-token rates (set LLM_INPUT_COST_PER_M / LLM_OUTPUT_COST_PER_M to match your model)"""
    input_cost_per_million = float(os.getenv("LLM_INPUT_COST_PER_M", "2.50"))
    output_cost_per_million = float(os.getenv("LLM_OUTPUT_COST_PER_M", "10.00"))

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million

    return input_cost + output_cost

def format_cost_display(input_tokens, output_tokens, cost, session_cost):
    """Format cost information for display"""
    return f"""
**Current Query:**
- Input tokens: {input_tokens:,}
- Output tokens: {output_tokens:,}
- Cost: ${cost:.6f}

**Session Total:**
- Total cost: ${session_cost:.6f}
"""

def hybrid_search(vector_store, question, keywords, num_chunks=20):
    """
    Hybrid search: combines semantic similarity with keyword filtering
    Returns chunks that either match semantically OR contain keywords
    """
    # Get semantic similarity results
    semantic_results = vector_store.similarity_search(question, k=num_chunks)

    # Get all documents and filter by keywords
    all_docs = vector_store.docstore._dict.values() if hasattr(vector_store.docstore, '_dict') else []

    keyword_results = []
    if keywords and all_docs:
        keywords_lower = [k.lower() for k in keywords]
        for doc in all_docs:
            doc_text_lower = doc.page_content.lower()
            if any(keyword in doc_text_lower for keyword in keywords_lower):
                keyword_results.append(doc)
                if len(keyword_results) >= num_chunks:
                    break

    # Combine results (deduplicate by content)
    seen_content = set()
    combined_results = []

    for doc in semantic_results + keyword_results:
        content_hash = hash(doc.page_content[:100])  # Hash first 100 chars for dedup
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            combined_results.append(doc)
            if len(combined_results) >= num_chunks * 1.5:  # Allow more results for hybrid
                break

    return combined_results

def get_selected_company_filings(ticker, selected_filing_labels):
    """Download only specific filings based on user selection"""
    if not selected_filing_labels:
        return None, None, None

    ticker = ticker.strip().upper()

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

        # Parse selected filings to extract accession numbers and form types
        # Format: "2024-01-15 | 10-K | 0001234567890"
        selected_accessions = []
        selected_forms = []
        selected_dates = []

        for label in selected_filing_labels:
            parts = label.split(" | ")
            if len(parts) == 3:
                date_part, form_part, accession_part = parts
                selected_dates.append(date_part.strip())
                selected_forms.append(form_part.strip())
                selected_accessions.append(accession_part.strip())

        if not selected_accessions:
            return None, None, None

        # Get submissions data to find primary document names
        SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        submissions_response = requests.get(
            SUBMISSIONS_URL,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()

        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        accessions_list = recent["accessionNumber"]
        primary_docs_list = recent["primaryDocument"]

        # Create mapping of accession -> primary document
        accession_to_primary_doc = {}
        for acc, primary_doc in zip(accessions_list, primary_docs_list):
            accession_to_primary_doc[acc] = primary_doc

        # Download and process each selected filing
        text_segments = []
        cik_nozero = cik_str.lstrip("0")

        for accession, form_type, date in zip(selected_accessions, selected_forms, selected_dates):
            # Get primary document filename
            primary_document = accession_to_primary_doc.get(accession)
            if not primary_document:
                print(f"Warning: Could not find primary document for {accession}")
                continue

            # Format accession for URL (remove dashes)
            accession_no_dash = accession.replace("-", "")

            # Use direct document URL (same as rag_engine.py)
            document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_dash}/{primary_document}"
            base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_dash}"

            print(f"Downloading {form_type} filing (accession: {accession})...")
            response = requests.get(document_url, headers={"User-Agent": USER_AGENT}, proxies=PROXIES, timeout=30)

            if response.status_code == 200:
                html_content = response.text

                print(f"  → HTML content size: {len(html_content):,} characters")

                # Check if this is just a cover page with exhibit references (common for 6-K, 8-K)
                if len(html_content) < 50000 and 'exhibit' in html_content.lower():
                    print(f"  [WARNING]  Detected cover page - looking for financial exhibits...")

                    # Parse to find exhibit links
                    from bs4 import BeautifulSoup
                    soup_check = BeautifulSoup(html_content, 'html.parser')

                    financial_exhibits = []
                    for link in soup_check.find_all('a', href=True):
                        href = link['href']
                        text = link.get_text(strip=True).lower()

                        # Look for financial statement exhibits
                        if any(keyword in text for keyword in ['financial statement', 'interim', 'earnings', 'results', 'condensed consolidated']):
                            if href.endswith('.htm') or href.endswith('.html'):
                                exhibit_url = f"{base_url}/{href}"
                                print(f"  → Found: {href}")
                                financial_exhibits.append((href, exhibit_url))

                    # Download first financial exhibit
                    if financial_exhibits:
                        exhibit_name, exhibit_url = financial_exhibits[0]
                        print(f"   Downloading exhibit: {exhibit_name}...")

                        try:
                            exhibit_response = requests.get(exhibit_url, headers={"User-Agent": USER_AGENT}, proxies=PROXIES, timeout=30)
                            exhibit_response.raise_for_status()

                            # Append exhibit content
                            exhibit_html = exhibit_response.text
                            html_content = html_content + "\n\n<!-- EXHIBIT -->\n\n" + exhibit_html
                            print(f"  [SUCCESS] Added exhibit ({len(exhibit_html):,} chars)")
                        except Exception as e:
                            print(f"  [WARNING]  Failed: {e}")

                # Extract text with page tracking (includes XBRL, tables, etc.)
                # Use the enhanced version with base_url and page tracking
                page_segments = extract_text_from_html_enhanced(
                    html_content,
                    base_url=base_url,
                    include_exhibits=False,
                    track_pages=True  # Enable page tracking
                )

                # Calculate total text size from all pages
                total_text = sum(len(page_text) for page_text, _, _ in page_segments)
                print(f"  → Extracted {len(page_segments)} pages, total size: {total_text:,} characters")

                # Debug: Check if XBRL data was extracted (check first page)
                if page_segments and "STRUCTURED FINANCIAL DATA (XBRL)" in page_segments[0][0]:
                    print(f"  [SUCCESS] XBRL data found")
                else:
                    print(f"  [WARNING]  No XBRL data found")

                # Debug: Check if financial statements were found
                combined_text = ' '.join(page_text for page_text, _, _ in page_segments)
                if "FINANCIAL STATEMENTS:" in combined_text:
                    print(f"  [SUCCESS] Financial statements found")
                else:
                    print(f"  [WARNING]  No financial statements found")

                # Debug: Check for GMV mentions (for Klarna test case)
                gmv_count = combined_text.upper().count("GMV")
                if gmv_count > 0:
                    print(f"  [SUCCESS] Found 'GMV' {gmv_count} times in text")
                else:
                    print(f"  [WARNING]  'GMV' not found in text")

                if page_segments:
                    print(f"  → Text preview: {page_segments[0][0][:500]}...")

                # Add page segments to text_segments with metadata
                for page_text, page_number, page_meta in page_segments:
                    text_segments.append((page_text, {
                        'form_type': form_type,
                        'filing_date': date,
                        'accession': accession,
                        'ticker': ticker,
                        'company_name': company_name,
                        'page_number': page_number  # Add actual page number
                    }))
            else:
                print(f"Warning: Failed to download filing {accession} - Status {response.status_code}")

        if not text_segments:
            return None, None, None

        # Create filing info string
        filing_info = f"{len(selected_accessions)} selected filings: {', '.join(set(selected_forms))} ({min(selected_dates)} to {max(selected_dates)})"

        return text_segments, company_name, filing_info

    except Exception as e:
        print(f"Error downloading selected filings: {e}")
        return None, None, None

def get_available_filings_list(ticker, form_types, start_date, end_date):
    """Get a list of available filings without downloading them"""
    if not ticker or not ticker.strip():
        return gr.CheckboxGroup.update(choices=[], value=[])

    if not form_types:
        return gr.CheckboxGroup.update(choices=[], value=[])

    ticker = ticker.strip().upper()

    # Extract form types from choices
    clean_form_types = []
    for form in form_types:
        if " [" in form:
            clean_form_types.append(form.split(" [")[0])
        elif " (" in form:
            clean_form_types.append(form.split(" (")[0])
        else:
            clean_form_types.append(form)

    try:
        # Get CIK from ticker
        tickerdict = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            proxies=PROXIES,
            timeout=10,
        ).json()

        cik_str = None
        for entry in tickerdict.values():
            if entry["ticker"].upper() == ticker.upper():
                cik_str = str(entry["cik_str"]).zfill(10)
                break

        if not cik_str:
            return gr.CheckboxGroup.update(choices=[f"[ERROR] Ticker {ticker} not found"], value=[])

        # Get submissions
        SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        submissions_response = requests.get(
            SUBMISSIONS_URL,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()

        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        accessions = recent["accessionNumber"]
        filing_dates = recent["filingDate"]

        # Filter by form types and date range
        matching_filings = []
        for idx, (form, date, accession) in enumerate(zip(forms, filing_dates, accessions)):
            if form.strip().upper() in clean_form_types and start_date <= date <= end_date:
                # Format: "2024-01-15 | 10-K | 0001234567890"
                filing_label = f"{date} | {form} | {accession}"
                matching_filings.append((filing_label, idx))

        if not matching_filings:
            return gr.CheckboxGroup.update(choices=["No matching filings found"], value=[])

        # Sort by date (most recent first)
        matching_filings.sort(reverse=True)

        choices = [label for label, idx in matching_filings]
        # Select all by default
        return gr.CheckboxGroup.update(choices=choices, value=choices)

    except Exception as e:
        return gr.CheckboxGroup.update(choices=[f"[ERROR] Error: {str(e)}"], value=[])

def extract_text_from_uploaded_files(files):
    """Extract text from uploaded PDF, DOCX, TXT, or HTML files"""
    if not files:
        return "", []

    extracted_texts = []
    file_names = []

    for file in files:
        try:
            file_path = file.name if hasattr(file, 'name') else file
            file_name = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()

            text = ""

            if file_ext == '.pdf':
                if PdfReader is None:
                    extracted_texts.append(f"[PDF Error: pypdf not installed for {file_name}]")
                    continue

                reader = PdfReader(file_path)
                text = "\n\n".join([page.extract_text() for page in reader.pages])

            elif file_ext in ['.docx', '.doc']:
                if Document is None:
                    extracted_texts.append(f"[DOCX Error: python-docx not installed for {file_name}]")
                    continue

                doc = Document(file_path)
                text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

            elif file_ext in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                    text = extract_text_from_html(html_content)
            else:
                extracted_texts.append(f"[Unsupported format: {file_name}]")
                continue

            if text.strip():
                extracted_texts.append(f"\n\n{'='*80}\n")
                extracted_texts.append(f"Document: {file_name}\n")
                extracted_texts.append(f"{'='*80}\n\n")
                extracted_texts.append(text)
                file_names.append(file_name)

        except Exception as e:
            extracted_texts.append(f"[Error extracting {file_name}: {str(e)}]")

    combined_text = "".join(extracted_texts)
    return combined_text, file_names

def get_available_form_types(ticker):
    """Get available form types for a given ticker from SEC"""
    if not ticker or not ticker.strip():
        return gr.CheckboxGroup.update(choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"], value=["10-K"])

    try:
        ticker = ticker.strip().upper()

        # Get CIK from ticker
        tickerdict = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            proxies=PROXIES,
            timeout=10,
        ).json()

        cik_str = None
        for entry in tickerdict.values():
            if entry["ticker"].upper() == ticker:
                cik_str = str(entry["cik_str"]).zfill(10)
                break

        if not cik_str:
            return gr.CheckboxGroup.update(
                choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                value=["10-K"],
                label=f"Form Types (Ticker '{ticker}' not found)"
            )

        # Get submissions
        SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        submissions_response = requests.get(
            SUBMISSIONS_URL,
            headers={"User-Agent": USER_AGENT},
            proxies=PROXIES,
            timeout=30
        )
        submissions_response.raise_for_status()
        submissions = submissions_response.json()

        # Get unique form types
        recent = submissions["filings"]["recent"]
        forms = recent["form"]

        unique_forms = {}
        for form in forms:
            clean_form = (form or "").strip().upper()
            if clean_form:
                unique_forms[clean_form] = unique_forms.get(clean_form, 0) + 1

        # Sort by count (most common first)
        sorted_forms = sorted(unique_forms.items(), key=lambda x: x[1], reverse=True)
        form_choices = [f"{form} ({count})" for form, count in sorted_forms[:20]]  # Top 20 forms

        # Pre-select common forms if available
        available_forms = [form for form, _ in sorted_forms]
        default_selection = []
        for common in ["10-K", "10-Q", "8-K"]:
            if common in available_forms:
                # Find the choice with this form type
                for choice in form_choices:
                    if choice.startswith(common + " "):
                        default_selection.append(choice)
                        break

        if not default_selection and form_choices:
            default_selection = [form_choices[0]]

        return gr.CheckboxGroup.update(
            choices=form_choices,
            value=default_selection,
            label=f"Form Types (Found {len(unique_forms)} types for {ticker})"
        )

    except Exception as e:
        return gr.CheckboxGroup.update(
            choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
            value=["10-K"],
            label=f"Form Types (Error fetching: {str(e)[:50]})"
        )

def get_available_form_types_multi(tickers_input):
    """Get available form types across multiple tickers"""
    if not tickers_input or not tickers_input.strip():
        return gr.CheckboxGroup.update(choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"], value=["10-K"])

    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        return gr.CheckboxGroup.update(choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"], value=["10-K"])

    try:
        # Get ticker to CIK mapping
        tickerdict = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            proxies=PROXIES,
            timeout=10,
        ).json()

        # Collect form types for all companies
        all_forms_by_ticker = {}  # {ticker: {form: count}}

        for ticker in tickers:
            # Find CIK
            cik_str = None
            for entry in tickerdict.values():
                if entry["ticker"].upper() == ticker:
                    cik_str = str(entry["cik_str"]).zfill(10)
                    break

            if not cik_str:
                continue

            # Get submissions
            SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
            try:
                submissions_response = requests.get(
                    SUBMISSIONS_URL,
                    headers={"User-Agent": USER_AGENT},
                    proxies=PROXIES,
                    timeout=30
                )
                submissions_response.raise_for_status()
                submissions = submissions_response.json()

                # Get unique form types for this ticker
                recent = submissions["filings"]["recent"]
                forms = recent["form"]

                unique_forms = {}
                for form in forms:
                    clean_form = (form or "").strip().upper()
                    if clean_form:
                        unique_forms[clean_form] = unique_forms.get(clean_form, 0) + 1

                all_forms_by_ticker[ticker] = unique_forms
            except:
                continue

        if not all_forms_by_ticker:
            return gr.CheckboxGroup.update(
                choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                value=["10-K"],
                label=f"Form Types (Could not fetch data for any ticker)"
            )

        # Aggregate forms across all tickers
        # Show form with list of tickers that have it
        all_forms = {}  # {form: [ticker1, ticker2, ...]}
        for ticker, forms_dict in all_forms_by_ticker.items():
            for form in forms_dict.keys():
                if form not in all_forms:
                    all_forms[form] = []
                all_forms[form].append(ticker)

        # Sort by number of companies that have this form (most common first)
        sorted_forms = sorted(all_forms.items(), key=lambda x: len(x[1]), reverse=True)

        # Create choices with ticker info - show ALL forms (not just top 25)
        form_choices = []
        for form, ticker_list in sorted_forms:  # Show ALL forms
            ticker_abbrev = ','.join(ticker_list[:3])  # Show first 3 tickers
            if len(ticker_list) > 3:
                ticker_abbrev += f",+{len(ticker_list)-3}"
            form_choices.append(f"{form} [{ticker_abbrev}]")

        # Pre-select common forms if available
        default_selection = []
        for common in ["10-K", "10-Q", "8-K"]:
            for choice in form_choices:
                if choice.startswith(common + " "):
                    default_selection.append(choice)
                    break

        if not default_selection and form_choices:
            default_selection = [form_choices[0]]

        company_count = len(all_forms_by_ticker)
        form_count = len(all_forms)

        return gr.CheckboxGroup.update(
            choices=form_choices,
            value=default_selection,
            label=f"Form Types (Found {form_count} types across {company_count} companies)"
        )

    except Exception as e:
        return gr.CheckboxGroup.update(
            choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
            value=["10-K"],
            label=f"Form Types (Error: {str(e)[:50]})"
        )

def get_cached_indexes_info():
    """Get information about cached indexes"""
    if not INDEX_DIR.exists():
        return "No cached indexes found."

    cached = []
    for metadata_file in INDEX_DIR.glob("*_metadata.pkl"):
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            index_key = metadata_file.stem.replace('_metadata', '')
            index_path = get_index_path(index_key)

            if index_path.exists():
                cached.append({
                    'Index Key': index_key[:16] + "...",  # Shortened for display
                    'Chunks': metadata.get('num_chunks', 'N/A'),
                    'Created': metadata.get('created_at', 'N/A')[:19],
                    'Size (MB)': round(index_path.stat().st_size / (1024*1024), 2)
                })
        except:
            continue

    if not cached:
        return "No cached indexes found."

    df = pd.DataFrame(cached)
    return df.to_string(index=False)

def prepare_document_with_metadata(text_segments, embeddings, index_key=None):
    """
    Prepare document with source metadata for each segment.
    text_segments: list of (text, metadata_dict) tuples
    """
    from langchain.schema import Document as LangChainDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("Preparing document with source tracking...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=400
    )

    all_docs = []
    chunk_global_index = 0  # Global chunk counter

    for doc_idx, (text, metadata) in enumerate(text_segments):
        # Split the text
        chunks = text_splitter.split_text(text)

        # Create Document objects with enhanced metadata
        for chunk_idx, chunk in enumerate(chunks):
            # Check if actual page number is available from metadata
            if 'page_number' in metadata:
                # Use actual page number from SEC HTML page breaks
                actual_page = metadata['page_number']
                page_type = 'actual'
            else:
                # Fallback: Estimate page number (for legacy data or non-SEC sources)
                char_position = chunk_idx * (8000 - 400)  # Account for overlap
                actual_page = (char_position // 3000) + 1
                page_type = 'estimated'

            # Add chunk metadata
            enhanced_metadata = metadata.copy()
            enhanced_metadata['chunk_index'] = chunk_global_index
            enhanced_metadata['chunk_in_doc'] = chunk_idx + 1
            enhanced_metadata['total_chunks_in_doc'] = len(chunks)
            enhanced_metadata['page_number'] = actual_page  # Actual or estimated
            enhanced_metadata['page_type'] = page_type  # Track if actual or estimated
            enhanced_metadata['estimated_page'] = actual_page  # Keep for backward compatibility
            enhanced_metadata['doc_index'] = doc_idx

            doc = LangChainDocument(page_content=chunk, metadata=enhanced_metadata)
            all_docs.append(doc)
            chunk_global_index += 1

    print(f"Created {len(all_docs)} chunks with metadata across {len(text_segments)} source documents")

    # Create FAISS vector store from documents
    print("Generating embeddings and building FAISS index...")
    vector_store = FAISS.from_documents(all_docs, embeddings)

    # Save to cache if provided
    if index_key:
        metadata_info = {
            'num_chunks': len(all_docs),
            'num_sources': len(text_segments),
            'chunk_size': 8000,
            'chunk_overlap': 400,
            'created_at': datetime.datetime.now().isoformat(),
            'metadata_version': '2.0',  # Track that this has enhanced chunk metadata
            'has_chunk_metadata': True
        }
        save_index(vector_store, index_key, metadata_info)

    return vector_store

def load_filings(ticker, form_types, start_date, end_date, uploaded_files, selected_filings=None, progress=gr.Progress()):
    """Load SEC filings with progress tracking, optionally including uploaded documents"""
    global vector_store, current_company, current_filing_info, embeddings, document_sources

    if not ticker and not uploaded_files:
        return "[ERROR] Please enter a ticker symbol or upload documents", "", ""

    if ticker and ticker.strip() and not form_types:
        return "[ERROR] Please select at least one form type", "", ""

    # Initialize AI if needed
    success, msg = initialize_ai()
    if not success:
        return msg, "", ""

    try:
        # Date inputs are already strings in YYYY-MM-DD format
        start_date_str = start_date
        end_date_str = end_date

        # Extract form types from choices (remove count/ticker info)
        # Format: "10-K (25)" or "10-K [AAPL,MSFT]"
        clean_form_types = []
        if form_types:
            for form in form_types:
                if " [" in form:
                    clean_form_types.append(form.split(" [")[0])
                elif " (" in form:
                    clean_form_types.append(form.split(" (")[0])
                else:
                    clean_form_types.append(form)
            form_types = clean_form_types

        # Process uploaded documents first
        uploaded_text = ""
        uploaded_file_names = []
        if uploaded_files:
            progress(0.1, desc=" Processing uploaded documents...")
            uploaded_text, uploaded_file_names = extract_text_from_uploaded_files(uploaded_files)
            if not uploaded_text:
                return "[ERROR] Failed to extract text from uploaded documents", "", ""

        # Download SEC filings if ticker provided
        sec_filing_text = ""
        company_name = None
        filing_info = ""
        cache_exists = False

        if ticker and ticker.strip():
            ticker = ticker.strip().upper()

            # Check cache status (cache includes uploaded docs in key)
            cache_suffix = f"_plus_{len(uploaded_files)}_docs" if uploaded_files else ""
            index_key = generate_index_key(ticker, form_types, start_date_str, end_date_str) + cache_suffix
            cache_exists = index_exists(index_key)

            cache_msg = "[SUCCESS] Pre-processed index found! Loading will be instant." if cache_exists else "[WARNING] No cache found. First-time processing will take several minutes."

            # Step 1: Download
            progress(0.2, desc=" Downloading filings from SEC EDGAR...")

            # Use selected filings if provided, otherwise download all filings in date range
            if selected_filings and len(selected_filings) > 0:
                print(f"Loading {len(selected_filings)} selected filings...")
                sec_text_segments, company_name, filing_info = get_selected_company_filings(ticker, selected_filings)
            else:
                print("Loading all filings in date range...")
                # NEW: get_company_filings in rag_engine returns text_segments (already processed)
                # Pass interactive=False to avoid prompts in GUI mode
                sec_text_segments, company_name, filing_info = get_company_filings(
                    ticker, form_types, start_date_str, end_date_str, interactive=False
                )

            if not sec_text_segments:
                if uploaded_text:
                    # Continue with just uploaded documents
                    sec_text_segments = []
                else:
                    return f"[ERROR] Failed to download filings for {ticker}", "", ""

            # Note: No need to call extract_text_from_html - rag_engine already does enhanced extraction
            # (includes XBRL, tables, financial statements, and optionally PDFs/OCR)
        else:
            # Only uploaded documents, no SEC filings
            index_key = f"uploaded_docs_{len(uploaded_files)}_{hash(uploaded_text[:1000])}"
            company_name = "Uploaded Documents"
            filing_info = "Uploaded Documents Only"
            cache_msg = ""

        # Prepare text segments with metadata for source tracking
        text_segments = []
        document_sources = []  # Track for UI

        # Add SEC filing segments (already processed with enhanced extraction)
        if sec_text_segments:
            text_segments.extend(sec_text_segments)
            # Track source for UI
            sec_form_types = filing_info.split("(")[0].strip() if "(" in filing_info else "SEC Filings"
            document_sources.append(f"SEC Filing: {sec_form_types}")

        # Add uploaded documents
        if uploaded_text and uploaded_file_names:
            text_segments.append((uploaded_text, {
                "source_type": "Uploaded Documents",
                "file_names": ", ".join(uploaded_file_names)
            }))
            document_sources.append(f"Uploaded: {', '.join(uploaded_file_names)}")

        if not text_segments:
            return "[ERROR] No content to process", "", ""

        # Step 3: Generate/load embeddings with metadata
        if cache_exists:
            progress(0.6, desc=" Loading pre-computed embeddings from cache...")
            vector_store = load_index(index_key, embeddings)[0]
        else:
            progress(0.6, desc=" Generating embeddings (this may take a few minutes)...")
            vector_store = prepare_document_with_metadata(text_segments, embeddings, index_key=index_key)
        current_company = company_name

        # Update filing info to include uploaded documents
        if uploaded_file_names:
            uploaded_info = f"Uploaded: {', '.join(uploaded_file_names)}"
            if filing_info:
                current_filing_info = f"{filing_info} + {uploaded_info}"
            else:
                current_filing_info = uploaded_info
        else:
            current_filing_info = filing_info

        progress(1.0, desc="[SUCCESS] Complete!")

        # Build status message
        status_parts = [f"[SUCCESS] Successfully loaded documents for **{company_name}**\n"]

        if filing_info and filing_info != "Uploaded Documents Only":
            status_parts.append(f" **SEC Filings:** {filing_info}\n")

        if uploaded_file_names:
            status_parts.append(f" **Uploaded Documents:** {', '.join(uploaded_file_names)}\n")

        if cache_msg:
            status_parts.append(f"\n{cache_msg}\n")

        status_parts.append("\nYou can now ask questions in the Q&A tab!")

        status_msg = "\n".join(status_parts)

        return status_msg, company_name, current_filing_info

    except Exception as e:
        return f"[ERROR] Error processing documents: {e}", "", ""

def analyze_text_with_calculation(vector_store, llm, user_question, document_filter="All Documents"):
    """Analyze text with step-by-step calculation mode with multi-query decomposition"""
    from langchain_core.messages import HumanMessage

    print(f"INFO: [CALCULATION MODE] Using multi-query strategy for comprehensive retrieval")
    print(f"INFO: Document filter set to: {document_filter}")

    # Step 1: Ask LLM to decompose the question into sub-queries
    decomposition_prompt = f"""You are helping with document retrieval. Break down this complex question into 2-5 simple, specific sub-questions that would help answer it completely.

Original question: {user_question}

Please provide ONLY a numbered list of sub-questions, one per line, without any additional explanation.
Format:
1. [specific sub-question 1]
2. [specific sub-question 2]
etc.

Sub-questions:"""

    print("\nDecomposing question into sub-queries...")
    decomp_output = llm.invoke([HumanMessage(content=decomposition_prompt)])
    sub_queries = [line.strip() for line in decomp_output.content.strip().split('\n') if line.strip() and line.strip()[0].isdigit()]

    print(f"Generated {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        print(f"  - {sq}")

    # Prepare metadata filter based on document_filter selection
    search_kwargs = {"k": 8}
    if document_filter and document_filter != "All Documents":
        # Extract the source type from the filter (e.g., "Uploaded: klarna_6k.pdf" -> "Uploaded")
        if document_filter.startswith("SEC Filing"):
            search_kwargs["filter"] = {"source_type": "SEC Filing"}
            print(f"  → Filtering for SEC Filings only")
        elif document_filter.startswith("Uploaded"):
            search_kwargs["filter"] = {"source_type": "Uploaded Documents"}
            print(f"  → Filtering for Uploaded Documents only")

    # Step 2: Retrieve chunks for each sub-query
    all_relevant_chunks = []
    seen_content = set()  # Avoid duplicate chunks

    for i, sub_query in enumerate(sub_queries):
        print(f"\nRetrieving for sub-query {i+1}: {sub_query[:60]}...")

        try:
            # Try with filter first
            results = vector_store.similarity_search(sub_query, **search_kwargs)
        except Exception as e:
            # Fallback to no filter if filtering not supported
            print(f"  Warning: Metadata filtering not available, searching all documents")
            results = vector_store.similarity_search(sub_query, k=8)

        for doc in results:
            content_hash = hash(doc.page_content[:200])  # Hash first 200 chars to detect duplicates
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_relevant_chunks.append(doc)

    print(f"\nTotal unique chunks retrieved: {len(all_relevant_chunks)}")

    result_docs = "\n\n".join([f'Chunk {i+1}: {doc.page_content}\n\n' for i, doc in enumerate(all_relevant_chunks)])

    # Print retrieved chunks to console for debugging
    print("\n" + "="*80)
    print("RETRIEVED CHUNKS FOR DEBUGGING:")
    print("="*80)
    for i, doc in enumerate(all_relevant_chunks[:5]):  # Show first 5 chunks
        preview = doc.page_content[:500] if len(doc.page_content) > 500 else doc.page_content
        print(f"\nChunk {i+1} (first 500 chars):\n{preview}...")
    print("="*80 + "\n")

    # Enhanced prompt for calculation mode
    calculation_prompt = f"""You are an experienced financial analyst with strong quantitative skills. Answer the question using the context provided from the documents.

IMPORTANT INSTRUCTIONS FOR CALCULATION MODE:
1. Show your work step-by-step
2. Cite specific documents and page numbers (if available) for each data point you extract
3. Number each step clearly (Step 1, Step 2, etc.)
4. For calculations, show the formula before computing the result
5. If you need data that isn't in the provided context, explicitly state what's missing
6. Double-check your math and reasoning
7. Present final answers clearly with appropriate units
8. SEARCH CAREFULLY through ALL provided chunks - the data points you need may be scattered across different chunks from different source documents

NOTE: The context below was retrieved using multiple targeted sub-queries to ensure comprehensive coverage across all source documents.

Question: {user_question}

Provided context from documents (retrieved via multi-query strategy):

{result_docs}

Please provide a detailed, step-by-step response following the instructions above."""

    llm_output = llm.invoke([HumanMessage(content=calculation_prompt)])

    print("\nCalculation Mode - Final Answer")
    print("=" * 80)
    print(llm_output.content)

    return llm_output.content, result_docs

def ask_question(question, num_chunks=20, keywords="", show_chunks=False):
    """Process a question about the loaded filings"""
    global vector_store, current_company, current_filing_info, qa_history, llm
    global session_input_tokens, session_output_tokens, session_total_cost

    if vector_store is None:
        return "[ERROR] Please load filings first using the 'Load Filings' tab", "", ""

    if not question or not question.strip():
        return "[ERROR] Please enter a question", "", ""

    try:
        print(f"Processing question: {question[:50]}...")
        print(f"Retrieving {num_chunks} chunks for analysis...")

        # Use hybrid search if keywords provided
        if keywords and keywords.strip():
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            print(f"Using hybrid search with keywords: {keyword_list}")

            # Get hybrid results
            results = hybrid_search(vector_store, question, keyword_list, num_chunks=int(num_chunks))

            # Format sources with page numbers
            formatted_sources = extract_source_information(results)

            # Format results for LLM context
            result_docs = "\n\n".join([
                f'Chunk {i+1} [from {doc.metadata.get("form_type", "Unknown")} filed {doc.metadata.get("filing_date", "Unknown")}]:\n{doc.page_content}\n'
                for i, doc in enumerate(results)
            ])

            print(f"Retrieved {len(results)} chunks (hybrid: semantic + keyword)")

            # Generate answer
            from langchain.schema import HumanMessage
            llm_output = llm.invoke([HumanMessage(content=f"""From the perspective of an experienced financial analyst, answer the question using the context provided from the SEC filings.

Question: {question}

Context from SEC filings:

{result_docs}

Please provide a comprehensive answer based on the context above. Pay special attention to mentions of: {', '.join(keyword_list)}""")])

            answer = llm_output.content
            sources = result_docs
        else:
            # Standard semantic search - use analyze_text_with_sources to get document objects
            answer, retrieved_docs = analyze_text_with_sources(vector_store, llm, question, num_chunks=int(num_chunks))

            # Format sources with page numbers
            formatted_sources = extract_source_information(retrieved_docs)

            # Keep original sources for token estimation
            sources = "\n\n".join([
                f'Chunk {i+1} [from {doc.metadata.get("form_type", "Unknown")} filed {doc.metadata.get("filing_date", "Unknown")}]:\n{doc.page_content}\n'
                for i, doc in enumerate(retrieved_docs)
            ])

        # Estimate tokens for cost tracking
        # Input: question + retrieved context (sources is already a string with all context)
        input_text = f"{question}\n\nContext:\n{sources}"
        input_tokens = estimate_tokens(input_text, model="claude")

        # Output: answer
        output_tokens = estimate_tokens(answer, model="claude")

        # Calculate costs
        query_cost = calculate_cost(input_tokens, output_tokens)

        # Update session totals
        session_input_tokens += input_tokens
        session_output_tokens += output_tokens
        session_total_cost += query_cost

        # Format cost display
        cost_display = format_cost_display(input_tokens, output_tokens, query_cost, session_total_cost)

        # Add to history
        qa_history.append({
            'Company': current_company,
            'Filing Info': current_filing_info,
            'Question': question,
            'Answer': answer,
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        search_method = "Hybrid (semantic + keyword)" if (keywords and keywords.strip()) else "Semantic similarity"
        result = f"""
**Question:** {question}

**Answer:** {answer}

---
**Sources:**
{formatted_sources if 'formatted_sources' in locals() else 'N/A'}

---
*Asked at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*Search method: {search_method}*
*Retrieved {num_chunks} chunks for analysis*
"""

        # Optionally show retrieved chunks for debugging
        if show_chunks:
            result += f"""

---

### Retrieved Chunks (Debug View):

{sources}
"""

        # Update history display
        history_df = pd.DataFrame(qa_history)
        history_text = ""
        for _, row in history_df.iterrows():
            history_text += f"\n{'='*80}\n"
            history_text += f"Company: {row['Company']}\n"
            history_text += f"Filing: {row['Filing Info']}\n"
            history_text += f"Time: {row['Timestamp']}\n"
            history_text += f"\nQ: {row['Question']}\n"
            history_text += f"\nA: {row['Answer']}\n"

        print("Question answered successfully!")
        return result, history_text, cost_display

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in ask_question: {error_details}")
        return f"[ERROR] Error generating answer: {str(e)}", "", ""

def clear_history():
    """Clear Q&A history and reset cost tracking"""
    global qa_history, session_input_tokens, session_output_tokens, session_total_cost
    qa_history = []
    session_input_tokens = 0
    session_output_tokens = 0
    session_total_cost = 0.0
    return "[SUCCESS] History cleared", "", ""

def export_qa_to_csv():
    """Export Q&A history to CSV"""
    global qa_history

    if not qa_history:
        return None

    df = pd.DataFrame(qa_history)
    output_path = f"SEC_QA_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    return output_path

def extract_source_information(retrieved_docs):
    """
    Extract source information from retrieved chunks, showing just page numbers.
    Groups by filing and lists page numbers.
    Returns a formatted string with source details.
    """
    if not retrieved_docs:
        return "No sources available"

    # Group pages by filing
    filing_pages = {}  # Key: (form_type, filing_date), Value: list of (page_num, page_type)

    for i, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        form_type = meta.get('form_type', 'Unknown')
        filing_date = meta.get('filing_date', 'Unknown')
        filing_key = (form_type, filing_date)

        # Extract page number
        if 'page_number' in meta:
            page_num = meta.get('page_number')
            page_type = meta.get('page_type', 'estimated')
        elif 'estimated_page' in meta:
            page_num = meta.get('estimated_page')
            page_type = 'estimated'
        else:
            page_num = None
            page_type = 'unknown'

        if filing_key not in filing_pages:
            filing_pages[filing_key] = []

        filing_pages[filing_key].append((page_num, page_type))

    # Format output by filing
    source_lines = []
    for (form_type, filing_date), pages in sorted(filing_pages.items()):
        # Get unique page numbers and determine if they're actual or estimated
        page_info = {}
        for page_num, page_type in pages:
            if page_num is not None and page_num != '?':
                if page_num not in page_info:
                    page_info[page_num] = page_type

        if page_info:
            # Sort page numbers
            sorted_pages = sorted(page_info.keys())

            # Determine predominant page type
            actual_count = sum(1 for pt in page_info.values() if pt == 'actual')
            estimated_count = sum(1 for pt in page_info.values() if pt == 'estimated')

            if actual_count > estimated_count:
                page_prefix = "pages"
            elif estimated_count > 0:
                page_prefix = "est. pages"
            else:
                page_prefix = "pages"

            # Format page list
            if len(sorted_pages) <= 5:
                page_list = ", ".join(str(p) for p in sorted_pages)
            else:
                page_list = f"{sorted_pages[0]}, {sorted_pages[1]}, ..., {sorted_pages[-1]} ({len(sorted_pages)} total)"

            source_lines.append(f"{form_type} filed {filing_date}: {page_prefix} {page_list}")
        else:
            # No page numbers available
            source_lines.append(f"{form_type} filed {filing_date}: page numbers not available")

    return "\n".join(source_lines) if source_lines else "No source information available"

def analyze_text_with_sources(vector_store, llm, user_question, num_chunks=10):
    """
    Query the vector store and get an answer with source document tracking.
    Returns: (answer, retrieved_docs)
    """
    from langchain_core.messages import HumanMessage

    # Retrieve relevant chunks
    results = vector_store.similarity_search(user_question, k=num_chunks)

    # Format context for LLM
    result_docs = "\n\n".join([
        f'Chunk {i+1} [from {doc.metadata.get("form_type", "Unknown")} filed {doc.metadata.get("filing_date", "Unknown")}]:\n{doc.page_content}\n'
        for i, doc in enumerate(results)
    ])

    # Generate answer
    llm_output = llm.invoke([HumanMessage(content=f"""From the perspective of an experienced financial analyst, answer the question using the context provided from the SEC filings.

Question: {user_question}

Context from SEC filings:

{result_docs}

Please provide a comprehensive answer based on the context above.""")])

    return llm_output.content, results  # Return actual document objects, not string

def generate_batch_summary_pdf(df, timestamp, questions):
    """Generate AI-powered summary PDF from batch Q&A results

    Args:
        df: DataFrame with batch Q&A results
        timestamp: Timestamp string for filename matching
        questions: List of questions asked

    Returns:
        tuple: (summary_md_path, summary_pdf_path) or (None, None) if failed
    """
    global llm

    try:
        # Prepare summary of results for AI analysis
        companies = df['Ticker'].unique().tolist()
        num_filings = len(df)
        successful = len(df[~df['Answer'].str.startswith('ERROR')])

        # Prepare all Q&A data for AI analysis
        summary_data = []
        for ticker in companies:
            company_data = df[df['Ticker'] == ticker]
            company_name = company_data['Company'].iloc[0] if len(company_data) > 0 else ticker

            # Get all answers for each unique filing
            unique_filings = company_data.drop_duplicates(subset=['Accession'])
            for _, row in unique_filings.iterrows():
                # Get all Q&A for this filing
                filing_qa = company_data[company_data['Accession'] == row['Accession']]
                for _, qa_row in filing_qa.iterrows():
                    answer_clean = qa_row['Answer'].replace('\n\n---\nSOURCES:', ' [SOURCES:')
                    sources = qa_row.get('Sources', 'N/A')
                    summary_data.append({
                        'ticker': ticker,
                        'company': company_name,
                        'form_type': qa_row['Form Type'],
                        'filing_date': qa_row['Filing Date'],
                        'accession': qa_row.get('Accession', 'N/A'),
                        'question': qa_row['Question'],
                        'answer': answer_clean[:1000],  # Truncate to fit within token limits with source citations
                        'sources': (sources[:100] + '...') if (isinstance(sources, str) and len(sources) > 100) else (sources if isinstance(sources, str) else 'N/A')
                    })

        # Create prompt for AI to generate summary
        questions_text = '\n'.join([f"- {q}" for q in questions])

        # Format data for AI - include ALL items from all companies
        data_text = ""

        # Use all data (no sampling)
        print(f"Including all {len(summary_data)} Q&A pairs in summary generation...")

        for item in summary_data:
            data_text += f"\n\n**{item['ticker']} ({item['company']}) - {item['form_type']} filed {item['filing_date']}**\n"
            data_text += f"**Accession:** {item['accession']}\n"
            data_text += f"**Q:** {item['question']}\n"
            data_text += f"**A:** {item['answer']}\n"
            # Include source reference (already truncated in data preparation)
            if item['sources'] != 'N/A':
                data_text += f"**Sources:** {item['sources']}\n"

        # Estimate token count (rough: 4 chars per token)
        estimated_tokens = len(data_text) // 4
        print(f"Data section: {len(data_text):,} characters (~{estimated_tokens:,} tokens)")
        if estimated_tokens > 180000:
            print(f"⚠️  WARNING: Approaching token limit! May need to reduce data or use a different approach.")

        prompt = f"""You are analyzing batch Q&A results from SEC filings. Create a comprehensive executive summary document in markdown format.

**Analysis Context:**
- Companies analyzed: {', '.join(companies)}
- Total filings processed: {num_filings}
- Questions asked:
{questions_text}

**IMPORTANT - Company Identification:**
Each Q&A result clearly identifies the ticker (AFRM, XYZ, SEZL, PYPL, KLAR) and company name. When analyzing data:
- **AFRM** = Affirm Holdings, Inc. (Affirm products only)
- **XYZ** = Block, Inc. (Square, Cash App, Afterpay products)
- **SEZL** = Sezzle Inc. (Sezzle products)
- **PYPL** = PayPal Holdings (PayPal products)
- **KLAR** = Klarna Group (Klarna products)

**CRITICAL:** Do NOT mix up products between companies. Each filing is labeled with its ticker - only attribute data to the correct company.

**Q&A Results:**
{data_text}

**Instructions:**
Generate a markdown document with the following structure (adapt based on the content):

1. **Title and Overview** - Brief introduction of what was analyzed
2. **Summary Comparison Table** - Create a table comparing key findings across companies (use ✅ Yes, ❌ No, ⚠️ Partial format where appropriate). Keep company names/tickers on single lines. VERIFY each company's products are correctly attributed.
3. **Key Insights** - 3-5 major findings or patterns observed. CITE SOURCES for each major finding (e.g., "10-Q filed 2026-02-05, Accession 0001628280-26-005855").
4. **Company-Specific Analysis** - Detailed findings for each company with specific quotes and page references where available. ONLY include data from that company's filings. Include source citations for all quantitative data and quotes.
5. **Limitations** - Note any gaps in the analysis

**Style Guidelines:**
- Use ticker symbols for brevity in tables
- **CRITICAL: Include source citations** for all quantitative data and key findings (use format: "Form Type filed Date, Accession Number, Page reference")
- When citing specific numbers or quotes, include the filing type, date, and accession number
- Highlight quantitative data and material impacts
- Use markdown formatting (headers, bold, lists, tables)
- Be concise but comprehensive
- DOUBLE-CHECK that products are attributed to the correct company

**Citation Format Examples:**
- "Delinquency rate of 1.2% for 30-59 days past due (10-Q filed 2026-02-05, Accession 0001628280-26-005855)"
- "Net charge-offs increased to $316.5M (10-Q filed 2026-02-05, est. page 66)"

Generate the complete markdown document now:"""

        # Call AI to generate summary
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_markdown = response.content

        # Save markdown file
        summary_md_path = f"Batch_QA_Summary_{timestamp}.md"
        with open(summary_md_path, 'w', encoding='utf-8') as f:
            f.write(summary_markdown)

        # Convert to PDF using pandoc
        summary_pdf_path = f"Batch_QA_Summary_{timestamp}.pdf"
        import subprocess
        result = subprocess.run(
            ['pandoc', summary_md_path, '-o', summary_pdf_path, '--latex-engine=xelatex', '-V', 'geometry:margin=0.75in'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Pandoc failed, return only markdown
            return summary_md_path, None

        return summary_md_path, summary_pdf_path

    except Exception as e:
        # If summary generation fails, don't break the batch analysis
        print(f"[WARNING] Summary PDF generation failed: {str(e)}")
        return None, None

def batch_qa_analysis(tickers_input, form_types, start_date, end_date, questions_input, force_recompute=False, progress=gr.Progress()):
    """Run batch Q&A analysis on multiple companies with multiple questions - processes each filing individually

    Args:
        force_recompute: If True, ignore cached embeddings and recompute from scratch (default: False)
    """
    global embeddings, llm

    # Initialize AI
    success, msg = initialize_ai()
    if not success:
        return msg, None

    if not tickers_input or not tickers_input.strip():
        return "[ERROR] Please enter at least one ticker", None

    if not form_types:
        return "[ERROR] Please select at least one form type", None

    if not questions_input or not questions_input.strip():
        return "[ERROR] Please enter at least one question", None

    # Parse inputs
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    questions = [q.strip() for q in questions_input.strip().split('\n') if q.strip()]

    # Extract form types from choices (remove count and ticker info)
    # Format: "10-K (25)" or "10-K [AAPL,MSFT]"
    clean_form_types = []
    for form in form_types:
        if " [" in form:
            clean_form_types.append(form.split(" [")[0])
        elif " (" in form:
            clean_form_types.append(form.split(" (")[0])
        else:
            clean_form_types.append(form)
    form_types = clean_form_types

    # Date inputs are already strings
    start_date_str = start_date
    end_date_str = end_date

    # Results storage
    all_results = []
    results_log = []

    import time
    batch_start_time = time.time()

    results_log.append(f"Batch Q&A Analysis - Individual Filing Mode")
    results_log.append(f"Companies: {', '.join(tickers)}")
    results_log.append(f"Form types: {', '.join(form_types)}")
    results_log.append(f"Date range: {start_date_str} to {end_date_str}")
    results_log.append(f"Questions per filing: {len(questions)}")
    results_log.append(f"Cache mode: {'FORCE RECOMPUTE (ignoring cache)' if force_recompute else 'USE CACHE (default)'}")
    results_log.append(f"\nQuestions to be asked:")
    for i, q in enumerate(questions, 1):
        results_log.append(f"  {i}. {q[:100]}{'...' if len(q) > 100 else ''}")
    results_log.append("\n" + "="*80 + "\n")

    # First pass: count total filings
    total_filings = 0
    ticker_filings_map = {}

    for ticker in tickers:
        try:
            # Get CIK from ticker
            results_log.append(f"Looking up ticker {ticker}...")
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
                results_log.append(f"[ERROR] Ticker {ticker} not found in SEC database")
                continue

            results_log.append(f"  Found: {company_name} (CIK: {cik_str})")

            # Get submissions
            SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
            results_log.append(f"  Fetching filings from SEC...")
            submissions_response = requests.get(
                SUBMISSIONS_URL,
                headers={"User-Agent": USER_AGENT},
                proxies=PROXIES,
                timeout=30
            )

            if submissions_response.status_code != 200:
                results_log.append(f"[ERROR] SEC API returned status {submissions_response.status_code} for {ticker}")
                continue

            submissions_response.raise_for_status()
            submissions = submissions_response.json()

            # Get filings from 'recent' section
            recent = submissions["filings"]["recent"]
            forms = list(recent["form"])
            accessions = list(recent["accessionNumber"])
            filing_dates = list(recent["filingDate"])
            primary_docs_list = list(recent["primaryDocument"])

            results_log.append(f"  Recent filings available: {len(forms)}")

            # Check if we need historical files (if date range is older than recent filings)
            if filing_dates and min(filing_dates) > start_date_str:
                results_log.append(f"  Recent filings start at {min(filing_dates)}, need to fetch historical data...")

                # Get historical filing files if they exist
                if "files" in submissions["filings"] and submissions["filings"]["files"]:
                    historical_files = submissions["filings"]["files"]
                    results_log.append(f"  Found {len(historical_files)} historical filing files")

                    # Only fetch historical files until we have data covering start_date
                    files_fetched = 0
                    for hist_file in historical_files:
                        hist_file_name = hist_file["name"]
                        hist_url = f"https://data.sec.gov/submissions/{hist_file_name}"

                        try:
                            hist_response = requests.get(
                                hist_url,
                                headers={"User-Agent": USER_AGENT},
                                proxies=PROXIES,
                                timeout=30
                            )
                            if hist_response.status_code == 200:
                                hist_data = hist_response.json()
                                # Append historical data to our lists
                                forms.extend(hist_data["form"])
                                accessions.extend(hist_data["accessionNumber"])
                                filing_dates.extend(hist_data["filingDate"])
                                primary_docs_list.extend(hist_data["primaryDocument"])

                                files_fetched += 1

                                # Check if we've reached far enough back in time
                                if filing_dates and min(filing_dates) <= start_date_str:
                                    results_log.append(f"  Fetched {files_fetched} historical files (reached {min(filing_dates)})")
                                    break

                        except Exception as e:
                            results_log.append(f"  Warning: Failed to fetch {hist_file_name}: {str(e)[:50]}")

                    results_log.append(f"  Total filings after adding historical: {len(forms)}")

            results_log.append(f"  Filtering for form types: {form_types}")
            results_log.append(f"  Date range: {start_date_str} to {end_date_str}")

            # Filter matching filings
            matching_filings = []
            form_types_upper = [ft.upper() for ft in form_types]

            for form, date, accession, primary_doc in zip(forms, filing_dates, accessions, primary_docs_list):
                if form.strip().upper() in form_types_upper and start_date_str <= date <= end_date_str:
                    matching_filings.append({
                        'form_type': form,
                        'filing_date': date,
                        'accession': accession,
                        'primary_document': primary_doc,
                        'company_name': company_name,
                        'ticker': ticker,
                        'cik_str': cik_str
                    })

            ticker_filings_map[ticker] = matching_filings
            total_filings += len(matching_filings)

            if len(matching_filings) == 0:
                results_log.append(f"[WARNING] No matching filings found for {ticker}")
                results_log.append(f"  Checked {len(forms)} total filings")
                # Show sample of what forms are available
                unique_forms = list(set(forms[:50]))[:10]  # First 10 unique form types
                results_log.append(f"  Available form types (sample): {', '.join(unique_forms)}")
            else:
                results_log.append(f"  Found {len(matching_filings)} matching filings")

        except Exception as e:
            results_log.append(f"[ERROR] Failed to get filings for {ticker}: {str(e)}")
            import traceback
            results_log.append(f"  Traceback: {traceback.format_exc()[:200]}")

    total_operations = total_filings * len(questions)
    results_log.append(f"Total filings to process: {total_filings}")
    results_log.append(f"Total Q&A analyses: {total_operations}")
    results_log.append("\n" + "="*80 + "\n")

    current_operation = 0

    # Process each ticker
    for ticker_idx, ticker in enumerate(tickers):
        filings_list = ticker_filings_map.get(ticker, [])

        if not filings_list:
            results_log.append(f"\n[{ticker_idx+1}/{len(tickers)}] {ticker}: No matching filings")
            continue

        results_log.append(f"\n[{ticker_idx+1}/{len(tickers)}] Processing {ticker} ({len(filings_list)} filings)...")

        # Process each filing individually
        for filing_idx, filing_info in enumerate(filings_list):
            filing_date = filing_info['filing_date']
            form_type = filing_info['form_type']
            accession = filing_info['accession']
            primary_document = filing_info['primary_document']
            company_name = filing_info['company_name']
            cik_str = filing_info['cik_str']

            results_log.append(f"  [{filing_idx+1}/{len(filings_list)}] {form_type} filed {filing_date}...")

            import time
            filing_start_time = time.time()

            try:
                # Generate unique index key for this specific filing
                filing_index_key = f"{ticker}_{form_type}_{filing_date}_{accession}"

                # Check if cache exists FIRST - skip download/extraction if cached (unless force_recompute)
                if not force_recompute and index_exists(filing_index_key):
                    results_log.append(f"    Cache status: HIT ({filing_index_key}) - loading from cache")
                    cache_start = time.time()
                    vector_store, cache_metadata = load_index(filing_index_key, embeddings)
                    cache_time = time.time() - cache_start
                    results_log.append(f"    ⏱ Cache load: {cache_time:.1f}s")

                    # Check if this is legacy cache without chunk metadata
                    if cache_metadata and not cache_metadata.get('has_chunk_metadata', False):
                        results_log.append(f"    [INFO] Legacy cache detected (no chunk metadata). Consider rebuilding for enhanced source tracking.")
                else:
                    if force_recompute and index_exists(filing_index_key):
                        results_log.append(f"    Cache status: IGNORED (force recompute enabled) - downloading and processing")
                    else:
                        results_log.append(f"    Cache status: MISS ({filing_index_key}) - downloading and processing")

                    # Download this specific filing
                    download_start = time.time()
                    cik_nozero = cik_str.lstrip("0")
                    accession_no_dash = accession.replace("-", "")
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_dash}/{primary_document}"
                    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{accession_no_dash}"

                    response = requests.get(document_url, headers={"User-Agent": USER_AGENT}, proxies=PROXIES, timeout=30)

                    if response.status_code != 200:
                        results_log.append(f"        [ERROR] Failed to download (status {response.status_code})")
                        for question in questions:
                            all_results.append({
                                'Company': company_name,
                                'Ticker': ticker,
                                'Form Type': form_type,
                                'Filing Date': filing_date,
                                'Accession': accession,
                                'Question': question,
                                'Answer': f'ERROR: Failed to download filing (HTTP {response.status_code})',
                                'Sources': 'N/A',
                                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        continue

                    html_content = response.text
                    download_time = time.time() - download_start
                    results_log.append(f"        ⏱ Download: {download_time:.1f}s")

                    # Check for cover page and look for exhibits (same logic as get_selected_company_filings)
                    if len(html_content) < 50000 and 'exhibit' in html_content.lower():
                        from bs4 import BeautifulSoup
                        soup_check = BeautifulSoup(html_content, 'html.parser')

                        financial_exhibits = []
                        for link in soup_check.find_all('a', href=True):
                            href = link['href']
                            text = link.get_text(strip=True).lower()

                            if any(keyword in text for keyword in ['financial statement', 'interim', 'earnings', 'results', 'condensed consolidated']):
                                if href.endswith('.htm') or href.endswith('.html'):
                                    exhibit_url = f"{base_url}/{href}"
                                    financial_exhibits.append((href, exhibit_url))

                        if financial_exhibits:
                            exhibit_name, exhibit_url = financial_exhibits[0]
                            try:
                                exhibit_response = requests.get(exhibit_url, headers={"User-Agent": USER_AGENT}, proxies=PROXIES, timeout=30)
                                exhibit_response.raise_for_status()
                                exhibit_html = exhibit_response.text
                                html_content = html_content + "\n\n<!-- EXHIBIT -->\n\n" + exhibit_html
                            except Exception as e:
                                pass  # Continue with just the cover page

                    # Extract text with page tracking (includes XBRL, tables, financial statements)
                    extract_start = time.time()
                    page_segments = extract_text_from_html_enhanced(
                        html_content,
                        base_url=base_url,
                        include_exhibits=False,
                        quick_mode=False,
                        track_pages=True  # Enable page tracking for actual page numbers
                    )
                    extract_time = time.time() - extract_start
                    results_log.append(f"        ⏱ Text extraction: {extract_time:.1f}s (full mode with XBRL + page tracking)")

                    if not page_segments or (len(page_segments) == 1 and len(page_segments[0][0]) < 100):
                        results_log.append(f"        [ERROR] No text extracted")
                        for question in questions:
                            all_results.append({
                                'Company': company_name,
                                'Ticker': ticker,
                                'Form Type': form_type,
                                'Filing Date': filing_date,
                                'Accession': accession,
                                'Question': question,
                                'Answer': 'ERROR: No text could be extracted from filing',
                                'Sources': 'N/A',
                                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        continue

                    # Transform page segments into text_segments format for prepare_document_with_metadata
                    # page_segments is a list of (page_text, page_number, page_metadata) tuples
                    text_segments = []
                    for page_text, page_number, page_meta in page_segments:
                        text_segments.append((page_text, {
                            'form_type': form_type,
                            'filing_date': filing_date,
                            'accession': accession,
                            'ticker': ticker,
                            'company_name': company_name,
                            'page_number': page_number  # Add actual page number
                        }))

                    # Prepare vector store for this single filing with enhanced metadata
                    embed_start = time.time()
                    vector_store = prepare_document_with_metadata(text_segments, embeddings, index_key=filing_index_key)
                    embed_time = time.time() - embed_start
                    results_log.append(f"        ⏱ Embedding/indexing: {embed_time:.1f}s")

                # Answer each question for this filing
                qa_start = time.time()
                for q_idx, question in enumerate(questions):
                    current_operation += 1
                    progress(current_operation / total_operations,
                            desc=f"{ticker} {form_type} {filing_date}: Q{q_idx+1}/{len(questions)}")

                    try:
                        # Use analyze_text_with_sources to get source information
                        answer, retrieved_docs = analyze_text_with_sources(vector_store, llm, question, num_chunks=10)

                        # Extract source information
                        sources = extract_source_information(retrieved_docs)

                        # Debug: Log if metadata is missing (only for first question)
                        if q_idx == 0 and retrieved_docs:
                            sample_meta = retrieved_docs[0].metadata
                            has_enhanced_meta = 'chunk_in_doc' in sample_meta
                            if not has_enhanced_meta:
                                results_log.append(f"    [INFO] Using legacy cache (no chunk metadata). Clear cache for enhanced source tracking.")

                        # Append sources to answer for easy mapping
                        answer_with_sources = f"{answer}\n\n---\nSOURCES: {sources}"

                        all_results.append({
                            'Company': company_name,
                            'Ticker': ticker,
                            'Form Type': form_type,
                            'Filing Date': filing_date,
                            'Accession': accession,
                            'Question': question,
                            'Answer': answer_with_sources,
                            'Sources': sources,  # Keep separate column for backward compatibility
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # Log completion after ALL questions are done
                        if q_idx == len(questions) - 1:
                            qa_time = time.time() - qa_start
                            results_log.append(f"    [OK] Answered {len(questions)} questions")
                            results_log.append(f"    ⏱ Q&A: {qa_time:.1f}s")
                            filing_total_time = time.time() - filing_start_time
                            results_log.append(f"    ⏱ TOTAL for this filing: {filing_total_time:.1f}s")

                    except Exception as e:
                        results_log.append(f"    [ERROR] Q{q_idx+1}: {str(e)[:50]}")
                        all_results.append({
                            'Company': company_name,
                            'Ticker': ticker,
                            'Form Type': form_type,
                            'Filing Date': filing_date,
                            'Accession': accession,
                            'Question': question,
                            'Answer': f'ERROR: {str(e)}',
                            'Sources': 'N/A',
                            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            except Exception as e:
                results_log.append(f"    [ERROR] Processing failed: {str(e)[:100]}")
                for question in questions:
                    all_results.append({
                        'Company': company_name,
                        'Ticker': ticker,
                        'Form Type': form_type,
                        'Filing Date': filing_date,
                        'Accession': accession,
                        'Question': question,
                        'Answer': f'ERROR: {str(e)}',
                        'Sources': 'N/A',
                        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

        results_log.append(f"  [SUCCESS] Completed {ticker}")

    # Summary
    batch_total_time = time.time() - batch_start_time
    results_log.append("\n" + "="*80)
    results_log.append(f"\nBATCH ANALYSIS COMPLETE!")
    results_log.append(f"Total filings processed: {total_filings}")
    results_log.append(f"Total Q&A pairs generated: {len(all_results)}")
    successful = len([r for r in all_results if not r['Answer'].startswith('ERROR')])
    results_log.append(f"Successful: {successful}")
    results_log.append(f"Failed: {len(all_results) - successful}")
    results_log.append(f"\n⏱ TIMING SUMMARY:")
    results_log.append(f"  Total time: {batch_total_time:.1f}s ({batch_total_time/60:.1f} minutes)")
    if total_filings > 0:
        results_log.append(f"  Average per filing: {batch_total_time/total_filings:.1f}s")
    if len(all_results) > 0:
        results_log.append(f"  Average per Q&A pair: {batch_total_time/len(all_results):.1f}s")

    # Create DataFrame for download
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"Batch_QA_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        results_log.append(f"\n[SUCCESS] Results saved to: {output_path}")

        # Generate AI-powered summary PDF
        results_log.append(f"\n[INFO] Generating AI-powered summary PDF...")
        summary_md, summary_pdf = generate_batch_summary_pdf(df, timestamp, questions)

        if summary_pdf:
            results_log.append(f"[SUCCESS] Summary PDF saved to: {summary_pdf}")
        elif summary_md:
            results_log.append(f"[WARNING] Summary markdown created ({summary_md}), but PDF generation failed")
        else:
            results_log.append(f"[WARNING] Summary generation failed")

        return "\n".join(results_log), output_path, summary_pdf
    else:
        return "\n".join(results_log) + "\n\n[ERROR] No results generated", None, None

def preprocess_filings(tickers_input, form_types, start_date, end_date, progress=gr.Progress()):
    """Pre-process multiple companies"""
    global embeddings

    # Initialize AI
    success, msg = initialize_ai()
    if not success:
        return msg

    if not tickers_input:
        return "[ERROR] Please enter at least one ticker"

    if not form_types:
        return "[ERROR] Please select at least one form type"

    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        return "[ERROR] No valid tickers provided"

    # Date inputs are already strings in YYYY-MM-DD format
    start_date_str = start_date
    end_date_str = end_date

    # Extract form types from choices (remove count/ticker info)
    # Format: "10-K (25)" or "10-K [AAPL,MSFT]"
    clean_form_types = []
    for form in form_types:
        if " [" in form:
            clean_form_types.append(form.split(" [")[0])
        elif " (" in form:
            clean_form_types.append(form.split(" (")[0])
        else:
            clean_form_types.append(form)
    form_types = clean_form_types

    results = []
    results.append(f"Pre-processing {len(tickers)} companies...")
    results.append(f"Companies: {', '.join(tickers)}")
    results.append(f"Forms: {', '.join(form_types)}")
    results.append(f"Date Range: {start_date_str} to {end_date_str}")
    results.append("\n" + "="*80 + "\n")

    successful = []
    failed = []

    for i, ticker in enumerate(tickers):
        progress((i + 0.5) / len(tickers), desc=f"Processing {ticker}...")
        results.append(f"\n[{i+1}/{len(tickers)}] Processing {ticker}...")

        try:
            index_key = generate_index_key(ticker, form_types, start_date_str, end_date_str)

            if index_exists(index_key):
                results.append(f"  [SUCCESS] {ticker}: Already cached, skipping")
                successful.append(ticker)
                continue

            # Download and extract (returns text_segments in rag_engine)
            results.append(f"   Downloading and processing filings...")
            text_segments, company_name, filing_info = get_company_filings(
                ticker, form_types, start_date_str, end_date_str, interactive=False
            )

            if not text_segments:
                results.append(f"  [ERROR] {ticker}: Failed to download")
                failed.append(ticker)
                continue

            # Generate embeddings (text already extracted with enhancements)
            results.append(f"   Generating embeddings...")
            vector_store = prepare_document(text_segments, embeddings, index_key=index_key)

            results.append(f"  [SUCCESS] {ticker}: Successfully pre-processed!")
            successful.append(ticker)

        except Exception as e:
            results.append(f"  [ERROR] {ticker}: Error - {e}")
            failed.append(ticker)

    # Summary
    results.append("\n" + "="*80)
    results.append("\n PRE-PROCESSING COMPLETE!\n")
    results.append(f"[SUCCESS] Successful: {len(successful)} - {', '.join(successful) if successful else 'None'}")
    if failed:
        results.append(f"[ERROR] Failed: {len(failed)} - {', '.join(failed)}")

    return "\n".join(results)

# Build Gradio Interface
with gr.Blocks(title="SEC Filing Analysis Tool", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # SEC Filing Analysis Tool
    """)

    with gr.Tabs():
        # Tab 1: Load Filings
        with gr.Tab(" Load Filings"):
            gr.Markdown("###  Step 1: Enter Ticker and Date Range")

            with gr.Row():
                with gr.Column():
                    ticker_input = gr.Textbox(
                        label="Ticker Symbol",
                        placeholder="e.g., AAPL",
                        value="AAPL"
                    )
                    form_types_input = gr.CheckboxGroup(
                        choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                        label="Form Types",
                        value=["10-K"]
                    )

                with gr.Column():
                    start_date_input = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value="2023-01-01",
                        placeholder="2023-01-01"
                    )
                    end_date_input = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.datetime.now().strftime("%Y-%m-%d"),
                        placeholder="2024-12-31"
                    )

            gr.Markdown("###  Step 2: Select Specific Filings (Optional)")
            gr.Markdown("Click below to see available filings and select which ones to include")
            find_filings_btn = gr.Button(" Find Available Filings", variant="secondary")

            selected_filings_input = gr.CheckboxGroup(
                label="Select Filings to Load",
                choices=[],
                value=[],
                info="All filings are selected by default. Uncheck any you don't want to include."
            )

            gr.Markdown("###  Step 3: Upload Supplemental Documents (Optional)")
            gr.Markdown("Upload additional documents (earnings supplements, presentations, etc.) to analyze alongside SEC filings")
            uploaded_files_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".html", ".htm"],
                type="file"
            )

            load_btn = gr.Button(" Load Selected Filings and Start Q&A", variant="primary", size="lg")
            load_output = gr.Markdown()

            # Hidden outputs for state
            company_state = gr.State()
            filing_info_state = gr.State()

            # Wire up ticker change to update form types
            ticker_input.change(
                fn=get_available_form_types,
                inputs=[ticker_input],
                outputs=[form_types_input]
            )

            # Wire up find filings button
            find_filings_btn.click(
                fn=get_available_filings_list,
                inputs=[ticker_input, form_types_input, start_date_input, end_date_input],
                outputs=[selected_filings_input]
            )

        # Tab 2: Ask Questions
        with gr.Tab(" Ask Questions"):
            gr.Markdown("### Ask Questions About Loaded Filings")

            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What are the major risk factors mentioned in the filings?",
                        lines=3
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        num_chunks_slider = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Number of chunks to retrieve",
                            info="Increase if answer seems incomplete or missing information"
                        )
                        keyword_input = gr.Textbox(
                            label="Force include keywords (optional)",
                            placeholder="e.g., tariff, tax, revenue (comma-separated)",
                            info="Hybrid search: retrieves chunks containing these keywords in addition to semantic matches"
                        )
                        show_chunks_checkbox = gr.Checkbox(
                            label="Show retrieved chunks (for debugging)",
                            value=False
                        )

                    ask_btn = gr.Button("Ask Question", variant="primary")
                    answer_output = gr.Markdown()

                with gr.Column(scale=2):
                    gr.Markdown("### Token Cost Estimator")
                    cost_output = gr.Markdown(
                        value="**Session Total:** $0.000000"
                    )

                    gr.Markdown("### Q&A History")
                    history_output = gr.Textbox(
                        label="Session History",
                        lines=15,
                        max_lines=25
                    )
                    with gr.Row():
                        clear_btn = gr.Button("Clear History")
                        export_btn = gr.Button("Export CSV")
                    csv_output = gr.File(label="Download CSV")

            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, num_chunks_slider, keyword_input, show_chunks_checkbox],
                outputs=[answer_output, history_output, cost_output]
            )

            clear_btn.click(
                fn=clear_history,
                inputs=[],
                outputs=[answer_output, history_output, cost_output]
            )

            export_btn.click(
                fn=export_qa_to_csv,
                inputs=[],
                outputs=[csv_output]
            )

        # Tab 3: View Cache
        with gr.Tab("️ View Cache"):
            gr.Markdown("### Cached Indexes")
            cache_output = gr.Textbox(label="Cached Indexes", lines=15)
            refresh_cache_btn = gr.Button(" Refresh Cache List")

            refresh_cache_btn.click(
                fn=get_cached_indexes_info,
                inputs=[],
                outputs=[cache_output]
            )

            # Auto-load on tab open
            demo.load(fn=get_cached_indexes_info, outputs=[cache_output])

        # Tab 4: Batch Q&A
        with gr.Tab(" Batch Q&A"):
            gr.Markdown("""
            ### Batch Q&A Analysis
            Analyze multiple companies with multiple questions at once.
            All results will be exported to a single CSV file.
            """)

            with gr.Row():
                with gr.Column():
                    batch_tickers = gr.Textbox(
                        label="Ticker Symbols (comma-separated)",
                        placeholder="e.g., AAPL,MSFT,GOOG",
                        value="AAPL"
                    )
                    batch_forms = gr.CheckboxGroup(
                        choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                        label="Form Types",
                        value=["10-K"]
                    )

                with gr.Column():
                    batch_start = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value="2023-01-01",
                        placeholder="2023-01-01"
                    )
                    batch_end = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.datetime.now().strftime("%Y-%m-%d"),
                        placeholder="2024-12-31"
                    )

            batch_questions = gr.Textbox(
                label="Questions (one per line)",
                placeholder="What are the major risk factors?\nSummarize the financial performance.\nWhat are the revenue trends?",
                lines=6
            )

            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                batch_force_recompute = gr.Checkbox(
                    label="Force Recompute Embeddings",
                    value=False,
                    info="Ignore cached embeddings and rebuild from scratch. Use this to upgrade old caches or change embedding parameters. ⚠️ Much slower!"
                )

            gr.Markdown("**Example:** Enter 2 companies and 3 questions = 6 Q&A pairs generated")

            batch_btn = gr.Button(" Run Batch Analysis", variant="primary", size="lg")
            batch_output = gr.Textbox(label="Analysis Log", lines=20, max_lines=30)
            batch_csv = gr.File(label="Download Results CSV")
            batch_pdf = gr.File(label="Download Summary PDF")

            # Wire up ticker change to update form types (for all tickers)
            batch_tickers.change(
                fn=get_available_form_types_multi,
                inputs=[batch_tickers],
                outputs=[batch_forms]
            )

            batch_btn.click(
                fn=batch_qa_analysis,
                inputs=[batch_tickers, batch_forms, batch_start, batch_end, batch_questions, batch_force_recompute],
                outputs=[batch_output, batch_csv, batch_pdf]
            )

        # Tab 5: Pre-process
        with gr.Tab("️ Pre-process Filings"):
            gr.Markdown("""
            ### Pre-process Filings
            Pre-compute embeddings for multiple companies to speed up future queries.
            """)

            with gr.Row():
                with gr.Column():
                    pp_tickers = gr.Textbox(
                        label="Ticker Symbols (comma-separated)",
                        placeholder="e.g., AAPL,MSFT,GOOG",
                        value="AAPL,MSFT"
                    )
                    pp_forms = gr.CheckboxGroup(
                        choices=["10-K", "10-Q", "8-K", "DEF 14A", "S-1"],
                        label="Form Types",
                        value=["10-K"]
                    )

                with gr.Column():
                    pp_start = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value="2023-01-01",
                        placeholder="2023-01-01"
                    )
                    pp_end = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.datetime.now().strftime("%Y-%m-%d"),
                        placeholder="2024-12-31"
                    )

            pp_btn = gr.Button(" Start Pre-processing", variant="primary", size="lg")
            pp_output = gr.Textbox(label="Processing Log", lines=20, max_lines=30)

            # Wire up ticker change to update form types (for all tickers)
            pp_tickers.change(
                fn=get_available_form_types_multi,
                inputs=[pp_tickers],
                outputs=[pp_forms]
            )

            pp_btn.click(
                fn=preprocess_filings,
                inputs=[pp_tickers, pp_forms, pp_start, pp_end],
                outputs=[pp_output]
            )

    # Wire up load_btn
    load_btn.click(
        fn=load_filings,
        inputs=[ticker_input, form_types_input, start_date_input, end_date_input, uploaded_files_input, selected_filings_input],
        outputs=[load_output, company_state, filing_info_state]
    )

if __name__ == "__main__":
    print("Starting SEC Filing Analysis GUI...")

    # Gradio is local-only by default; set SEC_RAG_SHARE=1 for a temporary public gradio.live link
    share = os.getenv("SEC_RAG_SHARE", "0") == "1"
    port = int(os.getenv("SEC_RAG_PORT", "7860"))

    print("Initializing AI components...")
    initialize_ai()

    print("\n" + "="*80)
    print("IMPORTANT: If the GUI doesn't open automatically, manually navigate to:")
    print(f"http://127.0.0.1:{port}")
    print("="*80 + "\n")

    # Enable queue for progress tracking with longer timeout
    demo.queue(
        concurrency_count=3,  # Allow 3 concurrent requests
        max_size=20,  # Queue size
        api_open=False  # Disable external API access checks
    )

    # Launch with inline error handling
    print("Launching Gradio interface...")
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=share,
            inbrowser=False,
            prevent_thread_lock=False,
            debug=False,
            show_error=False,
            _skip_check_connectivity=True  # Internal flag to skip connectivity check
        )
    except TypeError:
        # If _skip_check_connectivity doesn't exist in this version, try without it
        print("Trying without connectivity check flag...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=share,
            inbrowser=False,
            prevent_thread_lock=False,
            debug=False,
            show_error=False
        )


