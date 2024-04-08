# dictionary with all the related information for the review bombing incidents:
review_bombing_incidents = {
    "Skyrim-Paid-Mods": {
        "games_title_terms": ["*skyrim", "*Skyrim"],  # the glob terms to fetch the correct files from the local folder
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "23. April 2015 - Ende April 2015",  # only for display in Label Studio
        "rb_start_date": "23.04.2015",
        "rb_end_date": "01.05.2015",
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "14. Juni 2017 - Ende Juli 2017",
        "rb_start_date": "14.06.2017",
        "rb_end_date": "01.08.2017",
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "12. September 2017 - Anfang Oktober 2017",
        "rb_start_date": "12.09.2017",
        "rb_end_date": "13.10.2017",
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "Beginn 29.08.2017 (Fallout 4), bzw. 26.09.2017 (Skyrim Special Edition) - November 2017",
        "rb_start_date": "29.08.2017",
        "rb_end_date": "30.11.2017",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "September und Oktober 2018",
        "rb_start_date": "21.09.2018",
        "rb_end_date": "01.11.2018",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "28. Januar 2019 - Ende Februar 2019",
        "rb_start_date": "28.01.2019",
        "rb_end_date": "01.03.2019",  # only the time period for the other parts is considered here (i.e. the first RB)
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "03. April 2019 - Ende April 2019",
        "rb_start_date": "03.04.2019",
        "rb_end_date": "01.05.2019",
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "16. April 2019 - Mai 2019",
        "rb_start_date": "16.04.2019",
        "rb_end_date": "01.07.2019",  # relatively few reviews compared to others, so the time period is extended a bit
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "23. April 2019 - Ende Mai 2019",
        "rb_start_date": "23.04.2019",
        "rb_end_date": "01.06.2019",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "19. Oktober - Ende Oktober 2019",
        "rb_start_date": "19.10.2019",
        "rb_end_date": "01.11.2019",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "02. März - April 2020",
        "rb_start_date": "02.03.2020",
        "rb_end_date": "01.05.2020",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "21. Juli - Ende Juli 2021",
        "rb_start_date": "21.07.2021",
        "rb_end_date": "01.08.2021",
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
        "marked_steam_off_topic": True,
        "review_bomb_time": "24. Februar 2022 - Anfang April 2022",
        "rb_start_date": "24.02.2022",
        "rb_end_date": "13.04.2022",
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
        "marked_steam_off_topic": False,
        "review_bomb_time": "11. August 2023 - Mitte September 2023",
        "rb_start_date": "11.08.2023",
        "rb_end_date": "16.09.2023",
        "social_media_start_time": "04.10.2022",
        "social_media_end_time": "01.10.2023",
    },
}
