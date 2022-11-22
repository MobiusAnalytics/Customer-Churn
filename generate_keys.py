import pickle
from pathlib import Path
import streamlit_authenticator as stauth



names = ["Mobius DA"] #"Maria Jasmine",

usernames = ["mobiusanalytics"] #"mariajasmine",
password = ["@mOb071122dA"] #"mAria1@3",

hashed_passwords = stauth.Hasher(password).generate()

file_path = Path(__file__).parent/ "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords,file)