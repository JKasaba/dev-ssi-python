import pandas as pd
import io
import base64

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, 'Unsupported file format!'
    except Exception as e:
        return None, f'There was an error processing this file: {str(e)}'
    
    return df, None
