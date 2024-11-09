import pandas as pd
from io import BytesIO

# Create Excel from dictionary
def create_excel_from_dict(data):
    try:
        # Create a DataFrame for the main invoice details
        main_df = pd.DataFrame({k: [v] for k, v in data.items() if k != 'line_items'})

        # Create a DataFrame for line items if present
        if 'line_items' in data and isinstance(data['line_items'], list):
            line_items_df = pd.DataFrame(data['line_items'])
        else:
            line_items_df = pd.DataFrame()

        # Create an in-memory Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            main_df.to_excel(writer, sheet_name='Invoice Details', index=False)
            if not line_items_df.empty:
                line_items_df.to_excel(writer, sheet_name='Line Items', index=False)

        # Get the value of the in-memory Excel file
        excel_data = output.getvalue()
        return excel_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
