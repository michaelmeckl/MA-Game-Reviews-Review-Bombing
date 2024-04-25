import pandas as pd
from sklearn.preprocessing import MinMaxScaler


"""
Which numerical features are of importance?

- <u>Review Credibility Features:
</u> Text-Features (Länge, Profanity, Extreme Words, Rechtschreibfehler, Readability, Topics ?), Consistency Text mit 
Spielbeschreibung, Consistency Text / Sentiment mit Rating, (extremes) Rating / Consistency mit Average Rating, Spielzeit 
des Reviewers (insgesamt / zum Zeitpunkt des Reviews), (Spiel direkt über Steam gekauft), Zeitpunkt des Reviews, 
*(Helpfulness-Score?)*, Verhältnis Steam-Hilfreich zu    Lustig - Votes (mehr lustig als hilfreich ist vielleicht ein Indiz)

- <u>User Credibility Features</u>: 
Account-Erstelldatum, Expertise (Menge Reviews / Angesehen in Community / schon länger auf Plattform / ...), 
vorherige Reviews / Review Score Distribution, Anzahl geschriebener Reviews, Anzahl Spiele im Besitz, Anzahl Freunde

- <u>Overall Game Reviews Credibility Features</u>: 
Temporal Burst (Anzahl Reviews in best. kurzer Zeitspanne), Sentiment-Stabilität (Average Review-Sentiment in best. Zeitraum), 
Average Rating in Zeitraum, Anzahl sehr ähnlicher Reviews / Duplikate in best. Zeitraum, Veränderung in den Topics in best. Zeitraum
"""


def scale_numerical_features(df: pd.DataFrame, numerical_columns: list[str]):
    # use MinMaxScaler ? see https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
    # alternatively StandardScaler
    # or rescale additional features to the scale of bert embeddings ??
    pass

