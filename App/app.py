#Core Pkgs
import streamlit as st
import altair as alt
import plotly.express as px


#EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime
import time

#Utils
import joblib

pipe_lr=joblib.load(open("models/emotion_classifier_pipe_lr_full_03_December_2023.pkl","rb"))

# Track Utils
from dbTables import createPageVisitedTables,addPageVisitedDetails,viewAllPageVisitedDetails,addPredictionDetails,viewAllPredictionDetails,createEmotionTable

#fnx
def predict_emotions(docx):
	results=pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results=pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "joy":"🤗", "neutral":"😐",  "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
	st.title("Mood Mapper")
	menu=["Welcome","About Project","Home","Monitor","About US"]
	choice=st.sidebar.selectbox("Menu",menu)

	createPageVisitedTables()
	createEmotionTable()

	if choice=="About Project":
		st.header("About Mood Mapper")

		st.subheader("Introduction")
		st.write("The Application Mood Mapper is an web application that uses NLP to find emotion of user by taking the text sent by user over any communication medium such as Social media, Message Service etc. The application takes input from user analyzes it and provides the mood of user it also saves the mood of user in database to learn from it and train model for more accurate prediction.")


		st.subheader("About Natural Language Processing")
		st.write("Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans using natural language. The ultimate objective of NLP is to enable machines to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant")

		st.subheader("Importance of Emotion Detection")
		st.write("Emotion detection in text, also known as sentiment analysis, is pivotal across industries. It enables businesses to analyze customer feedback, manage brand reputation, and improve customer support by understanding and addressing emotions. This technology extends to market research, politics, healthcare, and employee engagement, offering insights into consumer preferences, public sentiment, mental health, and workplace dynamics. Emotion detection's impact is broad, influencing decision-making processes, user experiences, and personalized content delivery as technology evolves.")

		st.subheader("Challenges Faced in Emotion Detection")
		st.write("Emotion detection in text faces challenges such as subjectivity, ambiguity, and cultural variations. Multimodal inputs, imbalanced data, and privacy concerns pose additional hurdles. Temporal dynamics, linguistic variations, and the transferability of models across domains are key considerations. Limited context understanding and the demand for real-time processing further complicate accurate emotion detection. Addressing these challenges requires ongoing research, diverse datasets, and advancements in model sophistication to enhance the reliability and applicability of emotion detection systems")


		st.subheader("The following technologies were used to build this app:.")
		st.write("Streamlit- Streamlit is an open-source Python library that is used to create web applications for data science and machine learning projects with minimal effort. It is designed to make it easy for developers, especially those without web development experience, to create interactive and customizable web applications directly from Python scripts.")
		st.write("Spacy- is a popular and powerful open-source library for natural language processing tasks. It provides efficient and accurate tools for tasks such as tokenization, named entity recognition, part-of-speech tagging, and dependency parsing, and is well-suited for large-scale applications. With a user-friendly API and a growing community of contributors, Spacy has become a widely-used tool in academia, industry, and government.")
		st.write("Sublime Text- Sublime Text is a sophisticated text editor for code, markup, and prose. It's widely used by developers for its speed, ease of use, and powerful features. Here are some key aspects of Sublime Text")

		st.subheader("About the creators of app:")
		st.write("Mood Mapper was created by Pranav Deshmukh(12), Rushikesh Kadam(21), Prasanna Sawant(48) under the mentorship of Prof Monali Rajput")
		addPageVisitedDetails("About Project",datetime.now())

	elif choice=="Welcome":
		#Splash Screen
		st.image("Mood_Mapper.jpg",width=400)
		st.markdown(
		"""
		# Welcome to Mood Mapper
		An Application for predicting emotions in text.
		"""

		)
		placeholder=st.empty()

	

		placeholder.empty()

	elif choice == "Home":
	    	addPageVisitedDetails("Home", datetime.now())
	    	st.subheader("Home-Emotion in Text")

	    	with st.form(key='emotion_clf_form'):
	        	raw_text = st.text_area("Type Here")
	        	submit_text = st.form_submit_button(label='Submit')

	    	with st.form(key='emotion_clf_update_data'):
	        	selected_emotion = st.radio("Select the expected emotion:", ["anger", "disgust", "fear", "joy", "neutral", "sadness", "shame", "surprise"])
	        	submit_text2 = st.form_submit_button(label='Select Expected Value')

	    	if submit_text:
	    			col1,col2=st.columns(2)
	    			prediction=predict_emotions(raw_text)
	    			probability=get_prediction_proba(raw_text)
	    			addPredictionDetails(raw_text,prediction,np.max(probability),datetime.now())

	    			with col1:
	    				st.success("Original Text")
	    				st.write(raw_text)
	    				st.success("Prediction")
	    				emoji_icon=emotions_emoji_dict[prediction]
	    				st.write("{}:{}".format(prediction,emoji_icon))
	    				st.write("Confidence:{}%".format(np.round(np.max(probability)*100),2))

	    			with col2:
	    				st.success("Prediction Probability")
	    				#st.write(probability)
	    				proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
	    				#st.write(proba_df.T)
	    				proba_df_clean=proba_df.T.reset_index()
	    				proba_df_clean.columns=["emotions","probability"]
	    				fig=alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
	    				st.altair_chart(fig,use_container_width=True)

	    	if submit_text2:
	    		new_row = {'Emotion': selected_emotion, 'Text': raw_text}
	    		df = pd.read_csv(r'C:\Users\HP\Documents\JGitHub\sentimental_analysis\data\emotion_dataset_2.csv')
	    		df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
	    		df.to_csv(r'C:\Users\HP\Documents\JGitHub\sentimental_analysis\data\emotion_dataset_2.csv', index=False)
	    		st.success(f"Emotion '{selected_emotion}' added to the dataset.")

	    		
	elif choice=="Monitor":
		addPageVisitedDetails("Monitor",datetime.now())
		st.subheader("Monitor: Emotion in Text")

		with st.expander("Page Metrics"):
			pageVisitedDetails=pd.DataFrame(viewAllPageVisitedDetails(),columns=['Pagename','Time of Visit'])
			st.dataframe(pageVisitedDetails)

		pgCount=pageVisitedDetails['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
		c=alt.Chart(pgCount).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
		st.altair_chart(c,use_container_width=True)

		p=px.pie(pgCount,values='Counts',names='Pagename')
		st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			dfEmotions=pd.DataFrame(viewAllPredictionDetails(),columns=['RawText','Prediction','Probability','Time of Visit'])
			st.dataframe(dfEmotions)

		predictionCount=dfEmotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
		pc=alt.Chart(predictionCount).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
		st.altair_chart(pc,use_container_width=True)

	else:
		st.subheader("About Us: Emotion in Text")
		st.write("The Project was created by Pranav Deshmukh(12), Rushikesh Kadam(21), Prasanna Sawant(48) Under the guidance of Prof Monali Rajput")

		st.subheader("Pranav Deshmukh")
		st.write("Mr Pranav Deshmukh is MCA student at VESIT(Vivekanand Education Society Of Institute and Technolgy). He is Software Developer who has build several android and web apps. He has done bachelor's and HSC from K V Pendharkar College and SSC from New Lord English High School and is currently living at kalyan. Since from his school days he was so interested in coding so he took IT as second subject during his high school where he studied basics of web develoment which led to the foundation of his development studies during his bachelor's days he started exploring about android devlopment build some android applications and now is working as an freelancer.")

		st.subheader("Rushikesh Kadam")
		st.write("Mr Rushikesh Kadam is MCA student at VESIT(Vivekanand Education Society Of Institute and Technology). He has worked as an Associate Project Manager at Lionbridge Technologies & currently he is perceiving MCA. He has done his bachelor’s from Ruia College. A seasoned individual possessing excellent software development skills with demonstrated ability in handling software development lifecycle management related activities. Proven track record of collaborating with cross-functional teams, stakeholders, and clients to ensure seamless project execution and achieve desired outcomes. Having a professional working experience of 3.8+ years.")

		st.subheader("Prasanna Sawant")
		st.write ("Prasanna Sawant is a dynamic individual currently pursuing his Masters in Computer Application, showcasing a strong dedication to advancing his expertise in the field. Alongside his academic pursuits, he holds the role of Sort Incharge, demonstrating leadership and organizational skills. Prasanna's passion for technology is evident, as he actively engages with the latest advancements, showcasing a keen interest in staying at the forefront of the rapidly evolving tech landscape. With a blend of academic excellence and practical experience, Prasanna is poised to make significant contributions to the intersection of technology and application in the professional realm.")

		st.subheader("Monali Rajput")
		st.write("Assistant Professor Monali Rajput is a distinguished academician with a robust background in Information Technology, currently holding the position of Assistant Professor. Boasting a remarkable 14 years of experience in the field, Professor Rajput has earned her Bachelor's (B.E.) and Master's (M.E.) degrees in IT, establishing a solid foundation for her academic pursuits. Her expertise encompasses a diverse range of areas, including Information Security, Data Structures and Programming, Natural Language Programming, and Java Programming. Professor Rajput is not only recognized for her extensive knowledge but is also celebrated as one of the best mentors in her domain. Known for her exceptional dedication to her students, she transcends conventional teaching methods, actively engaging with her students and providing invaluable guidance. Her commitment to fostering a supportive learning environment is evident in her approach, where she is known for going above and beyond to ensure her students grasp complex concepts. Professor Monali Rajput's teaching style, marked by a genuine concern for her students' success, coupled with her unwavering support and availability for one-on-one consultations, sets her apart as an outstanding figure in the academic community.")

		addPageVisitedDetails("About US",datetime.now())


if __name__ =='__main__':
	main()
