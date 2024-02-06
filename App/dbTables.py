#Load Database Pkgs
import sqlite3
conn=sqlite3.connect('MoodMapper.db',check_same_thread=False)
c=conn.cursor()



#Fnx
def createPageVisitedTables():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTabl(pagename TEXT,timeOfvisit TIMESTAMP)')


def addPageVisitedDetails(pagename,timeOfVisit):
	c.execute('INSERT INTO pageTrackTabl(pagename,timeOfvisit) VALUES(?,?)',(pagename,timeOfVisit))
	conn.commit()

def viewAllPageVisitedDetails():
	c.execute('SELECT * FROM pageTrackTabl')
	data=c.fetchall()
	return data



#fnx To Track Input & Prediction

def createEmotionTable():
	c.execute('CREATE TABLE IF NOT EXISTS EmotionTable(rawtext TEXT,prediction TEXT,probability NUMBER,timeOfvisit TIMESTAMP)')


def addPredictionDetails(rawtext,prediction,probability,timeOfVisit):
	c.execute('INSERT INTO EmotionTable(rawtext,prediction,probability,timeOfvisit) VALUES(?,?,?,?)',(rawtext,prediction,probability,timeOfVisit))
	conn.commit()

def viewAllPredictionDetails():
	c.execute('SELECT * FROM EmotionTable')
	data=c.fetchall()
	return data

