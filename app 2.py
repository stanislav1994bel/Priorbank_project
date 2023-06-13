import sklearn


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import streamlit as st
import pickle
import numpy as np

import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('6.jpeg')


classifier_name=['LogisticRegression']
option = st.sidebar.selectbox('Алгоритм', classifier_name)
st.subheader(option)



#Importing model and label encoders
model=pickle.load(open("model.pkl","rb"))
#model_1 = pickle.load(open("final_rf_model.pkl","rb"))

#le_pik=pickle.load(open("label_encoding_for_gender.pkl","rb"))
#le1_pik=pickle.load(open("label_encoding_for_geo.pkl","rb"))


def predict_churn(id,age,car_own_flg,car_type_flg,appl_rej_cnt,good_work_flg,Score_bki,out_request_cnt,region_rating,home_address_cd,work_address_cd,income,SNA,first_time_cd,Air_flg,education_cd_ACD,education_cd_GRD,education_cd_PGR,education_cd_SCH,education_cd_UGR,gender_cd_F,gender_cd_M):
    input = np.array([[id,age,car_own_flg,car_type_flg,appl_rej_cnt,good_work_flg,Score_bki,out_request_cnt,region_rating,home_address_cd,work_address_cd,income,SNA,first_time_cd,Air_flg,education_cd_ACD,education_cd_GRD,education_cd_PGR,education_cd_SCH,education_cd_UGR,gender_cd_F,gender_cd_M]]).astype(np.float64)
    if option == 'LogisticRegression':
        prediction = model.predict_proba(input)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    #else:
        #red=0.30
        return float(pred)


def main():
    st.title("Кредитный скоринг")
    html_temp = """
    <div style="background-color:white ;padding:10px">
    <h2 style="color:red;text-align:center;">Заполни форму</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)





    st.sidebar.subheader("Приложение создано для развёртывания модели кредитного скоринга")
    st.sidebar.text("Разработчик - Белый С.А")


    id = st.slider("Баланс", 1, 2000000)
    age = st.slider("Возраст", 18, 63)
    car_own_flg = st.selectbox("Есть автомобиль?", ['0', '1'])
    Score_bki = st.slider('Скоринговый балл', 0, 400)

    

    car_type_flg = st.selectbox(" Автомобиль(иномарка)?", ['0', '1'])

    good_work_flg = st.selectbox("Есть хорошая работа ?", ['0', '1'])

    appl_rej_cnt = st.slider('Количество отказанных прошлых заявок', 0, 100)

    out_request_cnt = st.slider("Количество запросов в бюро", 0, 100)

    region_rating = st.slider('Рейтинг региона', 10, 200)

    home_address_cd = st.slider("Категоризатор домашнего адреса", 1, 10)

    work_address_cd = st.slider("Категоризатор рабочего адреса", 1, 10)

    income = st.slider("Зарплата", 0.00, 2000000.00)

    SNA = st.selectbox("Связь заявителя с клиентами", ['0', '1'])

    first_time_cd = st.selectbox("Давность наличия информации о заявителе", ['0', '1'])

    Air_flg = st.selectbox("Наличие загран паспорта", ['0', '1'])

    education_cd_SCH = st.selectbox("Среднее образование", ['0', '1'])

    education_cd_GRD = st.selectbox("Высшее образование", ['0', '1'])

    education_cd_PGR = st.selectbox("Окончил магистратуру", ['0', '1'])

    education_cd_UGR = st.selectbox("Окончил аспирантуру", ['0', '1'])

    education_cd_ACD = st.selectbox("Окончил докторантуру", ['0', '1'])

    gender_cd_F = st.selectbox("Женщина", ['0', '1'])

    gender_cd_M = st.selectbox("Мужчина", ['0', '1'])


    churn_html = """
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;"> Заявка отклонена</h2>
               </div>
            """
    no_churn_html = """
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;"> Кредит одобрен</h2>
               </div>
            """

    if st.button('Сделать прогноз'):
        output = predict_churn(id,age,car_own_flg,car_type_flg,appl_rej_cnt,good_work_flg,Score_bki,out_request_cnt,region_rating,home_address_cd,work_address_cd,income,SNA,first_time_cd,Air_flg,education_cd_ACD,education_cd_GRD,education_cd_PGR,education_cd_SCH,education_cd_UGR,gender_cd_F,gender_cd_M)
        st.success('Вероятность дефолта составляет {}'.format(output))
        

        if output > 0.6:
            st.markdown(churn_html, unsafe_allow_html= True)

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)

if __name__=='__main__':
    main()
