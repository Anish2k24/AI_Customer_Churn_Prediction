import pandas as pandas
import numpy as np
from sklearn.model_selection import train_test_split, cross_vas_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_and_prepare_data(filepath=)