from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sys
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

# Data Transformation Configuration Class
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_file_path = os.path.join("artifacts", "label_encoder.pkl")

# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Method to return preprocessor object
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            # Define the categorical columns (assuming all columns are categorical)
            categorical_columns = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
       'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
       'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
       'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
       'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
       'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
       'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
       'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
       'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
       'cramps', 'bruising', 'obesity', 'swollen_legs',
       'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
       'brittle_nails', 'swollen_extremeties', 'excessive_hunger','extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
       'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
       'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
       'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
       'stomach_bleeding', 'distention_of_abdomen',
       'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
       'prominent_veins_on_calf', 'palpitations', 'painful_walking',
       'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
       'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
       'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

            # Categorical pipeline creation
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='No')),
                ('onehotencoder', OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')),
                ('scalar', StandardScaler(with_mean=False))  # StandardScaler for numeric scaling
            ])

            # Column transformer to apply the categorical pipeline
            preprocessor = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])         
            logging.info("Data Transformation Pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in Data Transformation: {str(e)}")
            raise CustomException(e, sys)

    # Method to initiate data transformation process
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully read train and test data.")
            logging.debug(f"Train DataFrame Head: \n{train_df.head()}")
            logging.debug(f"Test DataFrame Head: \n{test_df.head()}")

            # Extracting preprocessor object
            preprocessor_obj = self.get_data_transformation_object()

            target_column = 'prognosis'

            # Splitting features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # Preprocessing target variable (label encoding)
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            # Applying transformations (fitting and transforming the training data)
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing to train and test datasets.")

            # Combining the transformed features with the target variables
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Saving the preprocessor object
            save_obj(file_path=self.data_transformation_config.preprocessor_ob_file_path, obj=preprocessor_obj)
            logging.info("Preprocessor object saved successfully.")

            # Saving the label encoder object
            save_obj(file_path=self.data_transformation_config.label_encoder_file_path, obj=label_encoder)
            logging.info("Label Encoder object saved successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path, self.data_transformation_config.label_encoder_file_path

        except Exception as e:
            logging.error(f"Error in initiating data transformation: {str(e)}")
            raise CustomException(e, sys)


