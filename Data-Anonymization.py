import pandas as pd
import numpy as np

class DataAnonymizationFriend:
    def __init__(self, input_file: str):
        self.input_file_friend = input_file
        self.data_friend = None

    def read_data_friend(self):
        self.data_friend = pd.read_csv(self.input_file_friend)  

def anonymize_numerical_columns_friend(self, privacy_level_friend: float):
    if privacy_level_friend > 5:
        raise ValueError("Privacy level cannot exceed 5.")
    elif privacy_level_friend < 1:
        raise ValueError("Privacy level cannot be less than 1.")
    
    scale = 1 / privacy_level_friend
    scale = min(scale, 0.2)  # Ensure the scale is at most 0.2 (equivalent to privacy level 5)
    
    for col in self.data_friend.select_dtypes(include=['number']):
        self.data_friend[col] += np.random.laplace(scale=scale, size=len(self.data_friend))
        self.data_friend[col] += np.random.laplace(scale=1/privacy_level_friend, size=len(self.data_friend))

    def anonymize_string_columns_friend(self, strategy_friend: str):
        for col in self.data_friend.select_dtypes(include=['object']):
            if strategy_friend == "suppression":
                self.data_friend[col] = "suppressed"
            elif strategy_friend == "generalization":
                self.data_friend[col] = self.data_friend[col].apply(lambda x: x.split()[0] if isinstance(x, str) else x)  # Example: Keep only the first word
            elif strategy_friend == "synthetic replacement":
                self.data_friend[col] = self.data_friend[col].apply(lambda x: x[::-1] if isinstance(x, str) else x)  # Example: Reverse the string
            else:
                raise ValueError("Invalid anonymization strategy. Please choose from Suppression, Generalization, or Synthetic Replacement.")

    def save_anonymized_data_friend(self, output_file_friend: str):
        self.data_friend.to_csv(output_file_friend, index=False)  

def main_friend():
    input_file_friend = "input.csv" 
    output_file_friend = "output2.csv"  

    anonymizer_friend = DataAnonymizationFriend(input_file_friend)
    anonymizer_friend.read_data_friend()

    privacy_level_friend = float(input("Enter privacy level for numerical columns: "))
    strategy_friend = input("Enter anonymization strategy for string columns (Suppression/Generalization/Synthetic Replacement): ").lower()

    anonymizer_friend.anonymize_numerical_columns_friend(privacy_level_friend)
    anonymizer_friend.anonymize_string_columns_friend(strategy_friend)
    anonymizer_friend.save_anonymized_data_friend(output_file_friend)

if __name__ == "__main__":
    main_friend()
