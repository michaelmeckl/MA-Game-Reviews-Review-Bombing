#!/usr/bin/python
# -*- coding: utf-8 -*-

import pathlib
import pprint
import re
import shutil
import pandas as pd
from sentiment_analysis_and_nlp.language_detection import detect_language
from utils.utils import enable_max_pandas_display_size, concat_generators, check_if_date_in_range

DATA_FOLDER = pathlib.Path(__file__).parent.parent / "data_for_analysis"
STEAM_DATA_FOLDER = DATA_FOLDER / "steam"
METACRITIC_DATA_FOLDER = DATA_FOLDER / "metacritic"
TWITTER_DATA_FOLDER = DATA_FOLDER / "tweets"
REDDIT_DATA_FOLDER = DATA_FOLDER / "reddit"

OUTPUT_FOLDER_REVIEWS = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "reviews"
OUTPUT_FOLDER_POSTS = pathlib.Path(__file__).parent.parent / "data_for_analysis_cleaned" / "posts"


###############################################################################
review_bombing_incidents = {
    "Skyrim-Paid-Mods": {
        "games_title_terms": ["*skyrim", "*Skyrim"],
        "social_media_title_terms": ["*Skyrim*"],
        "affected_games": "The Elder Scrolls V: Skyrim",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Die Entwickler "Bethesda" versuchten bezahlte Mods für Skyrim im Steam-Workshop '
                              'einzuführen, d.h. kostenpflichtige Zusatzinhalte, die direkt über Steam für das Spiel '
                              'heruntergeladen werden konnten. Allerdings sollten dabei nur 25 % der Einnahmen an '
                              'die Entwickler der Mods gehen und der Rest an Bethesda und Valve (der Firma hinter '
                              'Steam). Da das von vielen Spielern als Versuch der Entwickler gesehen wurde, '
                              'von der Arbeit der Modder selbst zu profitieren, und viele der Meinung '
                              'waren, Mods sollten grundsätzlich kostenlos sein, hinterließen einige negative '
                              'Reviews für Skyrim.',
        "review_bomb_time": "23. April 2015 - Ende April 2015",
        # the social media start and end times are a bit broader to also get a few posts before and after the actual
        # review bombing time period
        "social_media_start_time": "16.04.2015",
        "social_media_end_time": "01.07.2015",
    },
    "GrandTheftAutoV-OpenIV": {
        "games_title_terms": ["*grand-theft-auto-v", "*Grand_Theft_Auto_V"],
        "social_media_title_terms": ["*Grand Theft Auto V*"],
        "affected_games": "Grand Theft Auto V (GTA V)",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Der Publisher "Take-Two Interactive" forderte die Entwickler der beliebten '
                              'Modding-Plattform "OpenIV" in einer Unterlassungsaufforderung dazu auf, die Plattform '
                              'abzuschalten, um damit vor allem gegen potenziell bösartige oder betrügerische '
                              'Mehrspieler-Mods für Grand Theft Auto V vorzugehen. Diese Aktion führte zu einer '
                              'großen Menge an Kritik von Spielern sowie negativen Reviews für das Spiel GTA V, '
                              'da dadurch das Modding des Spiels und damit auch für viele der Spielspaß stark '
                              'eingeschränkt wurde.',
        "review_bomb_time": "14. Juni 2017 - Ende Juli 2017",
        "social_media_start_time": "07.06.2017",
        "social_media_end_time": "01.09.2017",
    },
    "Firewatch": {
        "games_title_terms": ["*firewatch"],
        "social_media_title_terms": ["*Firewatch*"],
        "affected_games": "Firewatch",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Die Entwickler "Campo Santo" reichten eine DMCA-Beschwerde gegen ein Video ein, '
                              'das der bekannte YouTuber "PewDiePie" zu Firewatch gemacht hatte (d.h. er wurde '
                              'gezwungen, das Video zu entfernen), nachdem PewDiePie während eines seiner Livestreams '
                              'eine rassistische Bemerkung geäußert hatte. Viele Spieler gaben daraufhin negative '
                              'Bewertungen zu Firewatch ab und bezeichneten die Entwickler als "Social Justice '
                              'Warrior" (SJW), oder behaupteten sie würden Zensur unterstützen.',
        "review_bomb_time": "12. September 2017 - Anfang Oktober 2017",
        "social_media_start_time": "05.09.2017",
        "social_media_end_time": "01.01.2018",
    },
    "Bethesda-Creation-Club": {
        "games_title_terms": ["*skyrim-special-edition", "*Skyrim_Special_Edition", "*fallout-4", "*Fallout_4"],
        "social_media_title_terms": ["*Bethesda Creation Club*", "*bethesda_creation_club*"],
        "affected_games": "Fallout 4 und The Elder Scrolls V: Skyrim Special Edition",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Der Grund war die Einführung von Bethesda\'s "Creation Club", einem direkt in das '
                              'Spiel eingebauten Shop für kuratierte Mods und Mikrotransaktionen, Ende August 2017 in'
                              ' Fallout 4 und einige Woche später in der Special Edition von Skyrim. Da einige der '
                              'dort angebotenen Inhalte bereits vorher kostenlos anderweitig verfügbar waren und , '
                              'das Ganze von vielen nur als einen Versuch der Entwickler gesehen wurde, um von der '
                              'Arbeit unabhängiger Modder selbst mit zu profitieren, führte das zu einer großen '
                              'Anzahl an negativen Reviews für die beiden Spiele.',
        "review_bomb_time": "Beginn 29.08.2017 (Fallout 4), bzw. 26.09.2017 (Skyrim Special Edition) - November 2017",
        "social_media_start_time": "22.08.2017",
        "social_media_end_time": "01.01.2018",
    },
    "TotalWar-Rome-II": {
        "games_title_terms": ["*total-war-rome-ii", "*Total_War_ROME_II_Emperor_Edition"],
        "social_media_title_terms": ["*Total War Rome II*"],
        "affected_games": "Total War: Rome II (& Emperor Edition)",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Das Spiel erhielt im März 2018 einen Patch, der die Wahrscheinlichkeit erhöhte, '
                              'dass im Spiel weibliche Generäle erscheinen konnten. Als im August ein Bild '
                              'eines Spielers auftauchte, dessen Armee nur von weiblichen Generälen angeführt '
                              'wurde, beschwerten sich einige Spieler über die historische Genauigkeit, '
                              'die gerade in einer Serie mit historischem Kontext wie Total War dadurch verloren '
                              'ginge. Daraufhin erklärte eine weibliche Community-Content-Editorin der '
                              'Entwickler "Creative Assembly", dass das Spiel nur "historisch authentisch, '
                              'aber nicht historisch genau" sein sollte und äußerte, dass Spieler, die das Spiel '
                              'nicht mögen, weil es im Spiel zu viele weibliche Charaktere gäbe, sowieso nicht die '
                              'Zielgruppe wären. Wenig überraschend führte das zu einem großen \'Shitstorm\' und einer '
                              'Menge negativer Reviews für das Spiel.',
        "review_bomb_time": "September und Oktober 2018",
        "social_media_start_time": "14.09.2018",
        "social_media_end_time": "01.01.2019",
    },
    "Metro-Epic-Exclusivity": {
        "games_title_terms": ["*metro*"],
        "social_media_title_terms": ["*Metro*"],
        "affected_games": '"Metro" - Serie: Metro 2033, Metro: Last Light, Metro 2033 Redux, Metro: Last Light Redux',
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Kurz vor Release von Metro Exodus wurde von Entwickler "4A Games" '
                              'und Publisher "Deep Silver" bekannt gegeben, dass Metro Exodus eine Zeit lang exklusiv '
                              'im neuen (und kontroversen) Epic Games Store angeboten werden würde, und nicht auf '
                              'anderen Plattformen wie Steam. Deshalb erhielten ältere Teile der Metro - Serie '
                              'auf Metacritic und vor allem auf Steam negative Reviews.',
        "review_bomb_time": "28. Januar 2019 - Ende Februar 2019",
        "social_media_start_time": "20.01.2019",
        "social_media_end_time": "01.03.2020",
    },
    "Borderlands-Epic-Exclusivity": {
        "games_title_terms": ["*borderlands*"],
        "social_media_title_terms": ["*Borderlands*"],
        "affected_games": '"Borderlands" - Serie: Borderlands (und GOTY - Editionen), Borderlands 2 und '
                          'Borderlands: The Pre-Sequel',
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Ein paar Monate vor Release von Borderlands 3 wurde von Entwickler "Gearbox Software" '
                              'und Publisher "2K" bekannt gegeben, dass Borderlands 3 eine Zeit lang exklusiv im '
                              'neuen (und kontroversen) Epic Games Store angeboten werden würde, und nicht auf '
                              'anderen Plattformen wie Steam. Deshalb erhielten ältere Teile der Borderlands - Serie '
                              'auf Steam und Metacritic negative Reviews.',
        "review_bomb_time": "03. April 2019 - Ende April 2019",
        "social_media_start_time": "20.03.2019",
        "social_media_end_time": "01.05.2020",
    },
    "Assassins-Creed-Unity": {
        "games_title_terms": ["*assassins-creed-unity", "*Assassin_s_Creed_Unity"],
        "social_media_title_terms": ["*Assassins Creed Unity*"],
        "affected_games": "Assassin's Creed Unity",
        "review_bomb_type": "positiv",
        "review_bomb_reason": 'Nach dem Brand in der Notre-Dame, die in Assassin\'s Creed Unity virtuell '
                              'nachgebildet ist, entschied Entwickler "Ubisoft" das Spiel daraufhin für eine Woche '
                              'kostenlos auf ihrer Uplay-Plattform zur Verfügung zu stellen und 500.000€ für den '
                              'Wiederaufbau der Notre-Dame zu spenden. Dies führte innerhalb kurzer Zeit zu einer '
                              'deutlich gestiegenen Zahl an Spielern und positiven Bewertungen.',
        "review_bomb_time": "17. April 2019 - Ende April 2019",
        "social_media_start_time": "15.04.2019",
        "social_media_end_time": "01.07.2019",
    },
    "Mortal-Kombat-11": {
        "games_title_terms": ["*mortal-kombat-11", "*Mortal_Kombat_11"],
        "social_media_title_terms": ["*Mortal Kombat 11*"],
        "affected_games": "Mortal Kombat 11",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Das Spiel erhielt zu Release sehr viele negative Bewertungen, unter anderem aufgrund '
                              'der von den Entwicklern "Netherrealm Studios" angeblich eingebauten "Social Justice '
                              'Warrior / SJW - Propaganda" und politischen Agenda (weil in einem der Enden des Spiels '
                              'ein dunkelhäutiger Charakter in die Vergangenheit reist, um die Sklaverei '
                              'abzuschaffen), vielen Mikrotransaktionen und dem Fehlen von beliebten Charakteren aus '
                              'früheren Teilen der Serie.',
        "review_bomb_time": "23. April 2019 - Ende Mai 2019",
        "social_media_start_time": "16.04.2019",
        "social_media_end_time": "01.07.2019",
    },
    "Crusader-Kings-II-Deus-Vult": {
        "games_title_terms": ["*crusader-kings-ii", "*Crusader_Kings_II"],
        "social_media_title_terms": [],
        "affected_games": "Crusader Kings II",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Ursache war die falsche Berichterstattung über ein Interview mit den Entwicklern '
                              'über den zukünftigen 3. Teil der "Crusader Kings" - Serie: laut der Berichterstattung '
                              'äußerten die Entwickler ("Paradox"), dass sie Begriffe wie "Deus Vult" aufgrund ihrer '
                              'rassistischen Konnotationen nicht in den neuen Teil, Crusader Kings III, '
                              'einbauen wollten (später stellte sich heraus, dass das so in dem Interview nie gesagt '
                              'wurde). Bei einer Spiele-Serie mit historischem Kontext, die auch andere kontroverse '
                              'Themen, u.a. Krieg oder Vergewaltigung, darstellte, kam das bei einigen in der '
                              'Community nicht gut an, weshalb bei dem damals aktuellen 2. Teil sehr viele negative '
                              'Reviews mit Bezug auf "Deus Vult" hinterlassen wurden.',
        "review_bomb_time": "19. Oktober - Ende Oktober 2019",
        "social_media_start_time": "12.10.2019",
        "social_media_end_time": "01.12.2019",
    },
    "The-Long-Dark-GeForce-Now": {
        "games_title_terms": ["*the-long-dark", "*The_Long_Dark"],
        "social_media_title_terms": [],
        "affected_games": "The Long Dark",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Die Entwickler ließen das Spiel von GeForce Now (Nvidia\'s Cloud Gaming - Plattform) '
                              'entfernen, weil Nvidia keine Erlaubnis des Entwicklers dafür hatte. Dadurch verloren '
                              'die Spieler, die das Spiel darüber gespielt hatten, den Zugriff auf das Spiel, '
                              'was dazu führte, dass die Entwickler wegen ihrer Entscheidung als gierig bezeichnet '
                              'wurden und negative Reviews für das Spiel hinterlassen wurden.',
        "review_bomb_time": "02. März - April 2020",
        "social_media_start_time": "01.03.2020",
        "social_media_end_time": "01.05.2020",
    },
    "Superhot-VR": {
        "games_title_terms": ["*superhot-vr", "*SUPERHOT_VR"],
        "social_media_title_terms": [],
        "affected_games": "Superhot VR",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Superhot VR erhielt ein Update, mit dem alle Szenen aus dem Spiel entfernt wurden, '
                              'in denen die Spielerfigur sich selbst verletzt oder tötet. Die Entwickler "Superhot '
                              'Team" erklärten, diese Szenen hätten "keinen Platz" im Spiel. Einige waren über das '
                              'Entfernen von Content enttäuscht und bezeichneten die Aktion als "Zensur" und '
                              '"verweichlicht" oder beschwerten sich über die angebliche "Wokeness" der Entwickler.',
        "review_bomb_time": "21. Juli - Ende Juli 2021",
        "social_media_start_time": "14.07.2021",
        "social_media_end_time": "01.09.2021",
    },
    "Ukraine-Russia-Conflict": {
        "games_title_terms": ["*cyberpunk-2077", "*cyberpunk-2077_03_2022", "*Cyberpunk_2077",
                              "*Cyberpunk_2077_03_2022", "*witcher*", "*frostpunk", "*S_T_A_L_K_E_R*",
                              "*s-t-a-l-k-e-r*"],
        "social_media_title_terms": ["*ukraine_russia_review_bombing*", "*comments_general_Cyberpunk 2077_2020*",
                                     "*submissions_general_Cyberpunk 2077_2020*", "*Cyberpunk 2077--query*",
                                     "*CDPR*", "*Frostpunk*", "*Gwent*", "*Witcher-*", "*tweets_russia_ukraine*",
                                     "*STALKER-*", "*Ukraine_Russia_ReviewBomb*"],
        "affected_games": '"Witcher" - Spiele: The Witcher 1 und 2 (und deren Enhanced Editionen), The Witcher 3 '
                          '(und Complete Edition), GWENT, Thronebreaker; Cyberpunk 2077; This War of Mine; Frostpunk; '
                          '"S.T.A.L.K.E.R." - Spiele',
        "review_bomb_type": "sowohl negativ als auch positiv",
        "review_bomb_reason": "Kurz nach Beginn des Russland-Ukraine-Krieges im Februar 2022 entschieden sich einige "
                              "Spieleentwickler dazu, sich klar gegen Russland zu positionieren, indem sie entweder "
                              "ihre Spiele nicht mehr in Russland zum Verkauf anboten, z.B. CD Projekt Red (CDPR), "
                              "ihre Spiele nicht mehr mit Rubel bezahlt werden konnten, z.B. die “S.T.A.L.K.E.R.” - "
                              "Serie, oder direkt Geld an die Ukraine spendeten (Entwickler von “This War of Mine” "
                              "und “Frostpunk”). Dadurch fühlten sich vor allem russische Spieler hintergangen oder "
                              "zu Unrecht bestraft. Aber auch andere Spieler beteiligten sich am Review Bombing, "
                              "da sie der Meinung waren, dass sich Spieleentwickler aus politischen Angelegenheiten "
                              "heraushalten sollten.",
        "review_bomb_time": "24. Februar 2022 - Anfang April 2022",
        "social_media_start_time": "17.02.2022",
        "social_media_end_time": "01.07.2022",
    },
    "Overwatch-2": {
        "games_title_terms": ["*overwatch-2", "*Overwatch_2"],
        "social_media_title_terms": ["*Overwatch 2*"],
        "affected_games": "Overwatch 2",
        "review_bomb_type": "negativ",
        "review_bomb_reason": 'Auslöser war der Steam-Release von Overwatch 2 (auf Metacritic wurde das Spiel schon '
                              '2022 veröffentlicht), bei dem das Spiel innerhalb von zwei Tagen zum am schlechtest '
                              'bewerteten Spiel auf Steam wurde. Overwatch 2 war für viele Spieler eine große '
                              'Enttäuschung und eine Ansammlung von gebrochenen Versprechen, da versprochene Inhalte '
                              'von Entwickler "Blizzard" nachträglich wieder storniert wurden, '
                              'viele Mikrotransaktionen und nur wenig neue Inhalte im Vergleich zum Vorgänger '
                              'enthalten waren. Zudem wurde mit dem Start von Overwatch 2 das erste Overwatch '
                              'abgeschaltet, was langjährige Spieler des ersten Teils verärgerte.',
        "review_bomb_time": "11. August 2023 - September 2023",
        "social_media_start_time": "04.10.2022",
        "social_media_end_time": "01.10.2023",
    },
}


# the keys in this dictionary must map to the file names in the Metacritic and Steam folders in
# 'data_for_analysis' without the underscores and hyphens (case doesn't matter)
game_names_mapping = {
    "assassins creed unity.csv": {
        "game": "Assassins Creed Unity",
        "game_name_display": "Assassin's Creed Unity",
    },
    "assassin s creed unity.csv": {
        "game": "Assassins Creed Unity",
        "game_name_display": "Assassin's Creed Unity",
    },
    "borderlands.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands",
    },
    "borderlands GOTY.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands GOTY",
    },
    "borderlands GOTY enhanced.csv": {
        "game": "Borderlands",
        "game_name_display": "Borderlands GOTY Enhanced",
    },
    "borderlands 2.csv": {
        "game": "Borderlands 2",
        "game_name_display": "Borderlands 2",
    },
    "borderlands the pre sequel.csv": {
        "game": "Borderlands: The Pre-Sequel",
        "game_name_display": "Borderlands: The Pre-Sequel",
    },
    "crusader kings ii.csv": {
        "game": "Crusader Kings II",
        "game_name_display": "Crusader Kings II",
    },
    # ! important to not use ".csv" here to find all the Cyberpunk 2077 files from different incidents
    "cyberpunk 2077": {
        "game": "Cyberpunk 2077",
        "game_name_display": "Cyberpunk 2077",
    },
    "fallout 4.csv": {
        "game": "Fallout 4",
        "game_name_display": "Fallout 4",
    },
    "firewatch.csv": {
        "game": "Firewatch",
        "game_name_display": "Firewatch",
    },
    "frostpunk.csv": {
        "game": "Frostpunk",
        "game_name_display": "Frostpunk",
    },
    "grand theft auto v.csv": {
        "game": "Grand Theft Auto V",
        "game_name_display": "Grand Theft Auto V",
    },
    "gwent the witcher card game.csv": {
        "game": "GWENT: The Witcher Card Game",
        "game_name_display": "GWENT: The Witcher Card Game",
    },
    "hogwarts legacy": {
        "game": "Hogwarts Legacy",
        "game_name_display": "Hogwarts Legacy",
    },
    "metro 2033.csv": {
        "game": "Metro 2033",
        "game_name_display": "Metro 2033",
    },
    "metro 2033 redux.csv": {
        "game": "Metro 2033 Redux",
        "game_name_display": "Metro 2033 Redux",
    },
    "metro last light.csv": {
        "game": "Metro: Last Light",
        "game_name_display": "Metro: Last Light",
    },
    "metro last light redux.csv": {
        "game": "Metro: Last Light Redux",
        "game_name_display": "Metro: Last Light Redux",
    },
    "mortal kombat 11.csv": {
        "game": "Mortal Kombat 11",
        "game_name_display": "Mortal Kombat 11",
    },
    "no mans sky.csv": {
        "game": "No Mans Sky",
        "game_name_display": "No Man's Sky",
    },
    "no man s sky.csv": {
        "game": "No Mans Sky",
        "game_name_display": "No Man's Sky",
    },
    "overwatch 2": {
        "game": "Overwatch 2",
        "game_name_display": "Overwatch 2",
    },
    "s t a l k e r shadow of chernobyl.csv": {
        "game": "STALKER: Shadow of Chernobyl",
        "game_name_display": "S.T.A.L.K.E.R.: Shadow of Chernobyl",
    },
    "s t a l k e r clear sky.csv": {
        "game": "STALKER: Clear Sky",
        "game_name_display": "S.T.A.L.K.E.R.: Clear Sky",
    },
    "s t a l k e r call of pripyat.csv": {
        "game": "STALKER: Call of Pripyat",
        "game_name_display": "S.T.A.L.K.E.R.: Call of Pripyat",
    },
    "superhot vr.csv": {
        "game": "Superhot VR",
        "game_name_display": "Superhot VR",
    },
    "the elder scrolls v skyrim.csv": {
        "game": "The Elder Scrolls V: Skyrim",
        "game_name_display": "The Elder Scrolls V: Skyrim",
    },
    "the elder scrolls v skyrim special edition.csv": {
        "game": "The Elder Scrolls V: Skyrim",
        "game_name_display": "The Elder Scrolls V: Skyrim Special Edition",
    },
    "the long dark.csv": {
        "game": "The Long Dark",
        "game_name_display": "The Long Dark",
    },
    "the witcher 2 assassins of kings.csv": {
        "game": "The Witcher 2: Assassins of Kings",
        "game_name_display": "The Witcher 2: Assassins of Kings",
    },
    "the witcher 2 assassins of kings enhanced edition.csv": {
        "game": "The Witcher 2: Assassins of Kings",
        "game_name_display": "The Witcher 2: Assassins of Kings Enhanced Edition",
    },
    "the witcher 3 wild hunt.csv": {
        "game": "The Witcher 3: Wild Hunt",
        "game_name_display": "The Witcher 3: Wild Hunt",
    },
    "the witcher 3 wild hunt complete edition.csv": {
        "game": "The Witcher 3: Wild Hunt",
        "game_name_display": "The Witcher 3: Wild Hunt - Complete Edition",
    },
    "the witcher.csv": {
        "game": "The Witcher",
        "game_name_display": "The Witcher",
    },
    "the witcher enhanced edition.csv": {
        "game": "The Witcher",
        "game_name_display": "The Witcher Enhanced Edition",
    },
    "thronebreaker the witcher tales.csv": {
        "game": "Thronebreaker: The Witcher Tales",
        "game_name_display": "Thronebreaker: The Witcher Tales",
    },
    "total war rome ii.csv": {
        "game": "Total War: ROME II",
        "game_name_display": "Total War: ROME II",
    },
    "total war rome ii emperor edition.csv": {
        "game": "Total War: ROME II",
        "game_name_display": "Total War: ROME II - Emperor Edition",
    },
}

###############################################################################


"""
def select_relevant_data_files(games_terms, select_only_reviews=False):
    # select all the relevant csv files for the given games with glob
    relevant_steam_files = []
    relevant_metacritic_files = []
    for game in games_terms:
        if select_only_reviews:
            pattern = f"*user_reviews*{game}*.csv"
        else:
            pattern = f"*{game}*.csv"
        steam_files = [f for f in STEAM_DATA_FOLDER.glob(pattern)]
        metacritic_files = [f for f in METACRITIC_DATA_FOLDER.glob(pattern)]
        relevant_steam_files.extend(steam_files)
        relevant_metacritic_files.extend(metacritic_files)

    print(f"{len(relevant_steam_files)} Steam files found:")
    pprint.pprint(relevant_steam_files)
    print(f"{len(relevant_metacritic_files)} Metacritic files found:")
    pprint.pprint(relevant_metacritic_files)
    return relevant_steam_files, relevant_metacritic_files
"""


def add_game_name_to_game_files():
    """
    Add two additional columns for the game name: "game_name_display" for the actual name that is shown in Label
    Studio and "game" for easier mapping (e.g. to map "Borderlands" and "Borderlands GOTY" to the same game).
    This way the game_info and user_reviews files in metacritic and steam can be matched and merged accordingly.
    Also add another column "source" to differentiate between the review platforms.
    """
    for file in concat_generators(METACRITIC_DATA_FOLDER.glob("*.csv"), STEAM_DATA_FOLDER.glob("*.csv")):
        # find the correct dict entry by checking if the filename contains the key string
        for key in game_names_mapping.keys():
            # unify file names by removing _, - or whitespace
            file_name_cleaned = re.sub(r'[_-]+', ' ', file.name).lower()
            if key.lower() in file_name_cleaned:
                df = pd.read_csv(file)
                if "source" not in df:
                    source_value = "Steam" if file.parent.name == "steam" else "Metacritic"
                    df.insert(0, "source", [source_value] * len(df))
                if "game_name_display" not in df:
                    game_name_display = game_names_mapping[key]["game_name_display"]
                    df.insert(0, "game_name_display", [game_name_display] * len(df))
                if "game" not in df:
                    game_name = game_names_mapping[key]["game"]
                    df.insert(0, "game", [game_name] * len(df))

                # overwrite the old file
                df.to_csv(file, index=False)


def add_rb_information_to_game_files(rb_incident_name):
    """
    Map each file in the steam and metacritic folder to the corresponding review bombing incident by adding
    additional columns for the rb incident (must be the same name as in the twitter and reddit files for mapping!).
    """
    print(f"\n##########################\nUpdating files for \"{rb_incident_name}\" ...\n")

    # create a subfolder for this review bombing incident if it doesn't exist yet
    Sub_Folder = OUTPUT_FOLDER_REVIEWS / rb_incident_name
    if not Sub_Folder.is_dir():
        Sub_Folder.mkdir()
    else:
        print("WARNING: Subfolder already exists!")
        answer = input(f"Do you want to overwrite the existing folder for \"{rb_incident_name}\"? [y/n]\n")
        if str.lower(answer) == "y" or str.lower(answer) == "yes":
            shutil.rmtree(Sub_Folder)
            Sub_Folder.mkdir()
        else:
            return

    rb_information = review_bombing_incidents[rb_incident_name]
    games_title_terms = rb_information["games_title_terms"]
    affected_games = rb_information["affected_games"]
    review_bomb_type = rb_information["review_bomb_type"]
    review_bomb_reason = rb_information["review_bomb_reason"]
    review_bomb_time = rb_information["review_bomb_time"]

    relevant_steam_files = []
    relevant_metacritic_files = []
    for term in games_title_terms:
        pattern = f"{term}.csv"
        steam_files = [f for f in STEAM_DATA_FOLDER.glob(pattern)]
        metacritic_files = [f for f in METACRITIC_DATA_FOLDER.glob(pattern)]
        relevant_steam_files.extend(steam_files)
        relevant_metacritic_files.extend(metacritic_files)

    print(f"{len(relevant_steam_files)} Steam files found:")
    pprint.pprint(relevant_steam_files)
    print(f"{len(relevant_metacritic_files)} Metacritic files found:")
    pprint.pprint(relevant_metacritic_files)

    # add the new columns and save updated data to new folder
    for file in relevant_steam_files + relevant_metacritic_files:
        df = pd.read_csv(file)
        df.insert(3, "review_bomb_reason", [review_bomb_reason] * len(df))
        df.insert(3, "review_bomb_time", [review_bomb_time] * len(df))
        df.insert(3, "review_bomb_type", [review_bomb_type] * len(df))
        df.insert(3, "affected_games", [affected_games] * len(df))
        df.insert(3, "review_bombing_incident", [rb_incident_name] * len(df))
        df.to_csv(Sub_Folder / f"{file.stem}_updated.csv", index=False)


def add_rb_information_to_social_media_files(rb_incident_name):
    """
    Map each file in the twitter and reddit folder to the corresponding review bombing incident.
    """
    print(f"\n##########################\nUpdating files for \"{rb_incident_name}\" ...\n")
    rb_information = review_bombing_incidents[rb_incident_name]
    social_media_title_terms = rb_information["social_media_title_terms"]
    affected_games = rb_information["affected_games"]

    relevant_twitter_files = []
    relevant_reddit_files = []
    for term in social_media_title_terms:
        pattern = f"{term}.csv"
        twitter_files = [f for f in TWITTER_DATA_FOLDER.glob(pattern)]
        reddit_files = [f for f in REDDIT_DATA_FOLDER.glob(pattern)]
        relevant_twitter_files.extend(twitter_files)
        relevant_reddit_files.extend(reddit_files)

    print(f"{len(relevant_twitter_files)} Twitter files found:")
    pprint.pprint(relevant_twitter_files)
    print(f"{len(relevant_reddit_files)} Reddit files found:")
    pprint.pprint(relevant_reddit_files)

    twitter_dataframes = []
    reddit_comments_dataframes = []
    reddit_submissions_dataframes = []
    # add the new columns and save updated dataframe to the correct list
    for file in relevant_twitter_files + relevant_reddit_files:
        df = pd.read_csv(file)
        df.insert(3, "affected_games", [affected_games] * len(df))
        df.insert(3, "review_bombing_incident", [rb_incident_name] * len(df))
        # add the social media data source (Twitter or Reddit) as an additional column
        if file.parent.name == "reddit":
            df.insert(3, "source", ["Reddit"] * len(df))
            if "comment" in file.name:
                reddit_comments_dataframes.append(df)
            else:
                reddit_submissions_dataframes.append(df)
        else:
            df.insert(3, "source", ["Twitter"] * len(df))
            twitter_dataframes.append(df)

    return twitter_dataframes, reddit_submissions_dataframes, reddit_comments_dataframes


def add_language_column(df: pd.DataFrame, content_column_names: list[str]):
    def check_language(post):
        text_content = post["combined_content"]
        # since it is only relevant whether it is in english or not other languages are ignored for now
        detected_language = "english" if detect_language(text_content) == "en" else "other"
        # detected_language = "english" if detect_contains_english(text_content) else "other"
        return detected_language

    # combine all the text column first (e.g. title and content for reddit submissions) into a new column with one
    # text divided by newline
    df["combined_content"] = df[[*content_column_names]].fillna('').apply("\n".join, axis=1)

    # add a new column to the given dataframe with the detected language of the text content
    lang_result = df.apply(lambda row: check_language(row), axis=1)
    df.insert(4, "detected_language", lang_result.reset_index(drop=True))
    # df.drop(columns=["combined_content"], axis=1, inplace=True)  # remove the temporarily created combined text column


def filter_post_time_period(df: pd.DataFrame, review_bombing_incident: str, is_reddit_submissions_file=False):
    # first, bring all date columns into the same format as the reviews have
    if is_reddit_submissions_file:
        df['created_at'] = pd.to_datetime(df['created_at'], unit="s")
        df.drop(columns=['created_at_formatted'], axis=1, inplace=True)  # not needed anymore
    else:
        df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.strftime('%d.%m.%Y %H:%M:%S')

    start_time = review_bombing_incidents[review_bombing_incident]["social_media_start_time"]
    end_time = review_bombing_incidents[review_bombing_incident]["social_media_end_time"]
    # add new column with True / False based on whether the post was created during the time of the review bomb
    in_rb_period_result = df["created_at"].apply(
        lambda post_date: check_if_date_in_range(post_date, start_time, end_time))
    df.insert(6, "in_rb_time_period", in_rb_period_result.reset_index(drop=True))


def map_combine_social_media_data():
    # map each file in the tweets folder to its corresponding review bombing incident by adding the relevant
    # dataframe columns and combine all twitter files per rb incident afterwards
    print("Mapping twitter files to corresponding review bombing incidents ...\n")

    for rb_name in review_bombing_incidents.keys():
        if rb_name in ["Crusader-Kings-II-Deus-Vult", "The-Long-Dark-GeForce-Now", "Superhot-VR"]:
            # no social media files were downloaded for these review bombing incidents as they are not used
            continue

        # create a subfolder for this review bombing incident if it doesn't exist yet
        Sub_Folder = OUTPUT_FOLDER_POSTS / rb_name
        if not Sub_Folder.is_dir():
            Sub_Folder.mkdir()
        else:
            print("WARNING: Subfolder already exists!")
            answer = input(f"Do you want to overwrite the existing folder for \"{rb_name}\"? [y/n]\n")
            if str.lower(answer) == "y" or str.lower(answer) == "yes":
                shutil.rmtree(Sub_Folder)
                Sub_Folder.mkdir()
            else:
                return

        twitter_df_list, reddit_submissions_df_list, reddit_comments_df_list = add_rb_information_to_social_media_files(rb_name)

        print("Combining files ...\n")
        twitter_df_combined = pd.concat(twitter_df_list)
        twitter_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        twitter_df_combined = twitter_df_combined.drop_duplicates(subset=['id']).reset_index(drop=True)

        # combine reddit comments and submissions into two different dataframes since they have different columns
        reddit_submissions_df_combined = pd.concat(reddit_submissions_df_list)
        reddit_submissions_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        reddit_submissions_df_combined = reddit_submissions_df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)

        reddit_comments_df_combined = pd.concat(reddit_comments_df_list)
        reddit_comments_df_combined.sort_values(by="created_at", ascending=False, inplace=True)
        reddit_comments_df_combined = reddit_comments_df_combined.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # add a new column that flags each posts as either english or not to make it easier to separate these later
        print("Detecting language for each post ...")
        add_language_column(twitter_df_combined, content_column_names=["content"])
        add_language_column(reddit_submissions_df_combined, content_column_names=["title", "content"])
        add_language_column(reddit_comments_df_combined, content_column_names=["content"])

        # Check for each post if it's creation time is approximately in the period of the review bombing incident
        print("Filtering time range ...")
        filter_post_time_period(twitter_df_combined, rb_name)
        filter_post_time_period(reddit_submissions_df_combined, rb_name, is_reddit_submissions_file=True)
        filter_post_time_period(reddit_comments_df_combined, rb_name)

        twitter_df_combined.to_csv(Sub_Folder / f"twitter_combined_{rb_name}.csv", index=False)
        reddit_submissions_df_combined.to_csv(Sub_Folder / f"reddit_submissions_combined_{rb_name}.csv", index=False)
        reddit_comments_df_combined.to_csv(Sub_Folder / f"reddit_comments_combined_{rb_name}.csv", index=False)
        print(f"Finished with review bombing incident {rb_name}\n")


if __name__ == "__main__":
    enable_max_pandas_display_size()

    if not OUTPUT_FOLDER_REVIEWS.is_dir():
        OUTPUT_FOLDER_REVIEWS.mkdir(parents=True)
    if not OUTPUT_FOLDER_POSTS.is_dir():
        OUTPUT_FOLDER_POSTS.mkdir(parents=True)

    add_game_name_columns = False
    if add_game_name_columns:
        add_game_name_to_game_files()

    add_review_bombing_information = False
    if add_review_bombing_information:
        # for rb_name in review_bombing_incidents:
        #     add_rb_information_to_game_files(rb_name)
        review_bombing_name = "Assassins-Creed-Unity"
        add_rb_information_to_game_files(review_bombing_name)

    map_social_media_data = False
    if map_social_media_data:
        map_combine_social_media_data()
