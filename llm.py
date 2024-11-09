import json
import streamlit as st
import openai

# Set API key for OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_llm(invoice_text):
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Extract and structure the following invoice data: {invoice_text}"}],
            functions=[
                {
                    "name": "structure_invoice",
                    "description": "Extract and structure invoice data into a standardized format",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {"type": "string", "description": "The name of the company issuing the invoice"},
                            "invoice_number": {"type": "string", "description": "The unique identifier for the invoice"},
                            "date": {"type": "string", "description": "The date of the invoice (ISO 8601 format preferred)"},
                            "due_date": {"type": "string", "description": "The due date for payment (ISO 8601 format preferred)"},
                            "total_amount": {"type": "number", "description": "The total amount due on the invoice"},
                            "currency": {"type": "string", "description": "The currency used in the invoice (e.g., USD, EUR)"},
                            "billing_address": {"type": "string", "description": "The billing address on the invoice"},
                            "line_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {"type": "string"},
                                        "quantity": {"type": "number"},
                                        "unit_price": {"type": "number"},
                                        "total": {"type": "number"}
                                    }
                                },
                                "description": "An array of items listed on the invoice"
                            }
                        },
                        "required": ["company_name", "invoice_number", "date", "total_amount"]
                    }
                }
            ],
            function_call={"name": "structure_invoice"}
        )

        print("Full API Response:", response)

        if response.choices and response.choices[0].message.function_call:
            try:
                result = json.loads(response.choices[0].message.function_call.arguments)
                print("LLM Result Type:", type(result))
                print("LLM Result Content:", result)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("The response was not in valid JSON format. Here's the raw text:")
                print(response.choices[0].message.function_call.arguments)
                return None
        else:
            print("No function call in the response")
            return None

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None
