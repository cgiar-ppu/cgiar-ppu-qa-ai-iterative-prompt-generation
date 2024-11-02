# logger.py

import threading
import pandas as pd

class ResultLogger:
    def __init__(self, output_csv):
        self.output_csv = output_csv
        self.lock = threading.Lock()
        # Initialize CSV file if it doesn't exist
        try:
            pd.read_csv(self.output_csv)
        except FileNotFoundError:
            pd.DataFrame(columns=[
                'result_code', 'prompt_id', 'model_name', 'perspective',
                'model_output', 'token_count', 'timestamp'
            ]).to_csv(self.output_csv, index=False)
    
    def log_result(self, result):
        with self.lock:
            df = pd.DataFrame([result])
            df.to_csv(self.output_csv, mode='a', header=False, index=False)
