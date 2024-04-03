# Überblick

Dieses Repository enthält den Code für meine Masterarbeit im Studiengang Medieninformatik an der Universität Regensburg,
bei der es um die Analyse und Erkennung des Phänomens "Review Bombing" bei Videospiel-Nutzerreviews geht. Dazu 
werden Videospiel-Reviews zusammen mit Metadaten und Nutzerinformationen für ausgewählte Spiele sowie Social Media - 
Daten aus den jeweiligen Zeiträumen der Review Bombing-Vorfälle heruntergeladen, analysiert und für die manuelle 
Annotation in Label Studio vorbereitet. Darauf aufbauend werden verschiedene DL-Modelle auf diesen Daten trainiert, 
mit dem Ziel die Zugehörigkeit eines Reviews zu einem Review Bombing automatisch zu bestimmen.

## Ordnerstruktur
* Die Skripte, um relevante Daten von Steam, Metacritic, Twitter / X und Reddit zu extrahieren befinden sich in den 
entsprechenden Ordnern: **Steam, Metacritic, Twitter, Reddit**
  * Genauere Informationen zu den extrahierten Review und Social Media - Daten sowie zum Code für das Herunterladen und 
  Extrahieren der Daten befindet sich in der [Data_Extraction.md](./Data_Extraction.md) - Datei. 
  In der [Twitter_Reddit_Search.md](./Twitter_Reddit_Search.md) - Datei befinden sich zudem einige Notizen zu den 
  Suchfunktionen der jeweiligen Seiten (Steam, Metacritic, Twitter / X und Reddit) und zur Begründung für die Auswahl 
  geeigneter Queries.
* Im [data_labeling_test](./data_labeling_test) - Ordner befindet sich ein (aufgegebener) Versuch mithilfe von 
  [Snorkel](https://www.snorkel.org/) die Reviews programmatisch basierend auf Heuristiken zu labeln.
* Der [sentiment_analysis_and_nlp](./sentiment_analysis_and_nlp) - Ordner enthält verschiedene NLP-Utilities (wie 
  Tokenization, Stopword Removal, Sentence Splitting, Spracherkennung, etc.) sowie ein paar Tests mit 
  Aspekt-basierter Sentiment-Analyse.
* Der [cleanup_analyze_data](./cleanup_analyze_data) - Ordner enthält den Code zur Analyse der Reviews und Social 
  Media - Daten sowie die Vorverarbeitung und Auswahl der Reviews für die Annotation in [Label Studio](https://labelstud.io/).
* Im [label_studio_study](./label_studio_study) - Ordner sind sowohl der Code für das Label Studio - SDK (welches 
  zum automatisierten Zuweisen und Exportieren der Reviews in Label Studio verwendet wurde) als auch die Analyse (+ 
  Kombination) der fertig annotierten Reviews und des Fragebogens enthalten.
* Im [classification](./classification) - Ordner befindet sich der Code für das Machine und Deep Learning.

## Requirements
Der gesamte Code wurde in Python 3.11 geschrieben. Die benötigten Python-Libraries befinden sich in der requirements.
txt. Unter anderem wurden BeautifulSoup zum Scraping, Pandas, Textblob, NLTK & Spacy zum Analysieren der Daten, 
Matplotlib und Seaborn für Visualisierungen sowie PyTorch und Hugging Face Transformers für das Deep Learning verwendet.

Für das Extrahieren der Daten werden plattformspezifische Credentials benötigt, die nicht im Repository enthalten sind:

#### Steam
Eine .env - Datei mit folgendem Aufbau wird im [Steam](./Steam) - Ordner erwartet:
```
STEAM_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

#### Reddit
Eine praw.ini - Datei mit folgendem Aufbau wird im [Reddit](./Reddit) - Ordner erwartet:
```ini
[Search_Reddit]
client_id=YOUR_APP_ID
client_secret=YOUR_APP_SECRET_KEY
username=YOUR_REDDIT_USERNAME
user_agent=script:Post_Comments_Search_Academic (by /u/%(username)s)
```

#### Twitter / X
Eine twitter_credentials.env - Datei mit folgendem Aufbau wird im [Twitter](./Twitter) - Ordner erwartet:
```
ACCOUNT_1_USERNAME = YOUR_TWITTER_ACCOUNT_USERNAME
ACCOUNT_1_PASSWORD = YOUR_TWITTER_ACCOUNT_PASSWORD

[optional:]
ACCOUNT_2_USERNAME = YOUR_SECOND_TWITTER_ACCOUNT_USERNAME
ACCOUNT_2_PASSWORD = YOUR_SECOND_TWITTER_ACCOUNT_PASSWORD
```


Da Metacritic keine API oder ähnliches zur Verfügung stellt, werden hierfür auch keine Zugangsdaten benötigt.

## Sonstiges
Die im Code referenzierten Ordner "data_for_analysis", "data_for_analysis_cleaned" und "data_for_labelstudio" befinden 
sich nicht mit im Github-Repository, sondern nur die finalen, annotierten Daten im [label_studio_study](./label_studio_study) - Ordner. 
Auch die *MA-Pre-Fragebogen.csv* - Datei im label_studio_study - Ordnern ist aus Datenschutzgründen nicht im 
Repository.
