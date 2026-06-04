import streamlit as st
import pickle
import pandas as pd

# def load_data():
#     with open("Salary_prediction.pkl","rb") as file:
#         return pickle.load(file)
def load_data():
    try:
        with open("Salary_prediction.pkl", "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Pickle loading error: {e}")
        raise 
        
data = load_data()
model = data["model"]
encoder = data["encode"]

st.title("Employee Salary Prediction System")

st.sidebar.header("""Predict the Salary based on:
- Experience
- Education
- Job role
- Age""")

job_role = ['Software Engineer', 'Data Analyst', 'Senior Manager',
       'Sales Associate', 'Director', 'Marketing Analyst',
       'Product Manager', 'Sales Manager', 'Marketing Coordinator',
       'Senior Scientist', 'Software Developer', 'HR Manager',
       'Financial Analyst', 'Project Manager', 'Customer Service Rep',
       'Operations Manager', 'Marketing Manager', 'Senior Engineer',
       'Data Entry Clerk', 'Sales Director', 'Business Analyst',
       'VP of Operations', 'IT Support', 'Recruiter', 'Financial Manager',
       'Social Media Specialist', 'Software Manager', 'Junior Developer',
       'Senior Consultant', 'Product Designer', 'CEO', 'Accountant',
       'Data Scientist', 'Marketing Specialist', 'Technical Writer',
       'HR Generalist', 'Project Engineer', 'Customer Success Rep',
       'Sales Executive', 'UX Designer', 'Operations Director',
       'Network Engineer', 'Administrative Assistant',
       'Strategy Consultant', 'Copywriter', 'Account Manager',
       'Director of Marketing', 'Help Desk Analyst',
       'Customer Service Manager', 'Business Intelligence Analyst',
       'Event Coordinator', 'VP of Finance', 'Graphic Designer',
       'UX Researcher', 'Social Media Manager', 'Director of Operations',
       'Senior Data Scientist', 'Junior Accountant',
       'Digital Marketing Manager', 'IT Manager',
       'Customer Service Representative', 'Business Development Manager',
       'Senior Financial Analyst', 'Web Developer', 'Research Director',
       'Technical Support Specialist', 'Creative Director',
       'Senior Software Engineer', 'Human Resources Director',
       'Content Marketing Manager', 'Technical Recruiter',
       'Sales Representative', 'Chief Technology Officer',
       'Junior Designer', 'Financial Advisor', 'Junior Account Manager',
       'Senior Project Manager', 'Principal Scientist',
       'Supply Chain Manager', 'Senior Marketing Manager',
       'Training Specialist', 'Research Scientist',
       'Junior Software Developer', 'Public Relations Manager',
       'Operations Analyst', 'Product Marketing Manager',
       'Senior HR Manager', 'Junior Web Developer',
       'Senior Project Coordinator', 'Chief Data Officer',
       'Digital Content Producer', 'IT Support Specialist',
       'Senior Marketing Analyst', 'Customer Success Manager',
       'Senior Graphic Designer', 'Software Project Manager',
       'Supply Chain Analyst', 'Senior Business Analyst',
       'Junior Marketing Analyst', 'Office Manager', 'Principal Engineer',
       'Junior HR Generalist', 'Senior Product Manager',
       'Junior Operations Analyst', 'Senior HR Generalist',
       'Sales Operations Manager', 'Senior Software Developer',
       'Junior Web Designer', 'Senior Training Specialist',
       'Senior Research Scientist', 'Junior Sales Representative',
       'Junior Marketing Manager', 'Junior Data Analyst',
       'Senior Product Marketing Manager', 'Junior Business Analyst',
       'Senior Sales Manager', 'Junior Marketing Specialist',
       'Junior Project Manager', 'Senior Accountant', 'Director of Sales',
       'Junior Recruiter', 'Senior Business Development Manager',
       'Senior Product Designer', 'Junior Customer Support Specialist',
       'Senior IT Support Specialist', 'Junior Financial Analyst',
       'Senior Operations Manager', 'Director of Human Resources',
       'Junior Software Engineer', 'Senior Sales Representative',
       'Director of Product Management', 'Junior Copywriter',
       'Senior Marketing Coordinator', 'Senior Human Resources Manager',
       'Junior Business Development Associate', 'Senior Account Manager',
       'Senior Researcher', 'Junior HR Coordinator',
       'Director of Finance', 'Junior Marketing Coordinator',
       'Junior Data Scientist', 'Senior Operations Analyst',
       'Senior Human Resources Coordinator', 'Senior UX Designer',
       'Junior Product Manager', 'Senior Marketing Specialist',
       'Senior IT Project Manager', 'Senior Quality Assurance Analyst',
       'Director of Sales and Marketing', 'Senior Account Executive',
       'Director of Business Development', 'Junior Social Media Manager',
       'Senior Human Resources Specialist', 'Senior Data Analyst',
       'Director of Human Capital', 'Junior Advertising Coordinator',
       'Junior UX Designer', 'Senior Marketing Director',
       'Senior IT Consultant', 'Senior Financial Advisor',
       'Junior Business Operations Analyst',
       'Junior Social Media Specialist',
       'Senior Product Development Manager', 'Junior Operations Manager',
       'Senior Software Architect', 'Junior Research Scientist',
       'Senior Financial Manager', 'Senior HR Specialist',
       'Senior Data Engineer', 'Junior Operations Coordinator',
       'Director of HR', 'Senior Operations Coordinator',
       'Junior Financial Advisor', 'Director of Engineering',
       'Software Engineer Manager', 'Back end Developer',
       'Senior Project Engineer', 'Full Stack Engineer',
       'Front end Developer', 'Front End Developer',
       'Director of Data Science', 'Human Resources Coordinator',
       'Junior Sales Associate', 'Human Resources Manager',
       'Juniour HR Generalist', 'Juniour HR Coordinator',
       'Digital Marketing Specialist', 'Receptionist',
       'Marketing Director', 'Social Media Man', 'Delivery Driver']

col1,col2 = st.columns(2)

with col1:
    name = st.text_input("Name", placeholder = "Enter your name")
    # age = st.sidebar.slider("Age",18,60,25)
    age = st.text_input(" Age",placeholder = "Enter your age ")
    gender = st.selectbox("Gender",["Male","Female"])
    education = st.selectbox("Your Education",["High School","Bachelor's", "Master's", "PhD"])

with col2:
    job = st.selectbox("Search your job_role",
                       options = sorted(job_role),
                       index = None,
                        placeholder = "Type to search your job role..")
    experience = st.text_input("Work experience", placeholder = "Enter your work experience")

col1,col2,col3 = st.columns(3)
with col2:
    predict = st.button("Predict Your Salary", type = "primary")

if predict:
    user_input = {
        'Gender': gender,
        'Education Level': education, 
        'Job Title': job, 
        'Years of Experience': experience
    }
    input_df = pd.DataFrame([user_input])


    try:
        for col in encoder.keys():
            input_df[col] = encoder[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Salary: {prediction:,.0f}")

    except Exception as e:
        st.exception(f"an error accures during the prediction{e}")
