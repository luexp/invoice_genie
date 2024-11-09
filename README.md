# Invoice Analyzer

This project is an AI-powered invoice analysis tool that extracts and structures data from uploaded invoices using OCR, transformer models, and LLM processing.

## Features

- Upload invoice images
- Extract text and tables using EasyOCR
- Pass extracted data to GPT-4 to structure invoice data
- Process invoice data with the Donut transformer
- Compare the results of the Donut transformer with the LLM output for higher accuracy
- Display results in a user-friendly Streamlit interface
- Generate downloadable Excel reports

## Components

### app.py

The main Streamlit application that handles:
- User interface
- File uploads
- API requests to the backend
- Result display and visualization

### llm.py

Manages the interaction with the OpenAI GPT-4 model:
- Structures the extracted invoice data
- Defines the function call for invoice data extraction

### excel_creation.py

Handles the creation of Excel reports:
- Converts structured invoice data into Excel format
- Creates separate sheets for invoice details and line items
