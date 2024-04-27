#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools
import pprint
import numpy as np
import pathlib
import pandas as pd
import simpledorff
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score
# import json


DATA_FOLDER = pathlib.Path(__file__).parent / "exported_data" / "reviewed"
OUTPUT_FOLDER = pathlib.Path(__file__).parent / "parsed_data"
PLOT_FOLDER = pathlib.Path(__file__).parent / "plots"

# maps the label studio projects to the corresponding review bombing incidents
project_rb_incident_mapping = {
    53222: "Skyrim - Paid Mods",
    53221: "Assassin's Creed Unity",
    52584: "Firewatch",
    52581: "Mortal Kombat 11",
    52580: "Borderlands - Epic Store",
    52575: "Ukraine-Russia-Support",
}

# the column names of the annotation results
annotation_questions = ["is-review-bombing", "is-rating-game-related", "criticism-praise-game-content",
                        "criticism-praise-developer-publisher", "criticism-praise-ideological",
                        "criticism-praise-political"]


def calculate_cappa_score_for_prestudy():
    # TODO use fleiss's kappa instead?
    pre_study_df = pd.read_csv(pathlib.Path(__file__).parent / "exported_data" / "exported_tasks_52140_Vorstudie.csv")
    pre_study_df = pre_study_df.replace({'no': 'Nein', 'yes': 'Ja'})
    annotator_1 = pre_study_df[pre_study_df["annotator"] == 17120]
    annotator_2 = pre_study_df[pre_study_df["annotator"] == 17156]

    kappa_scores = []
    prestudy_annotation_questions = ["review-bombing", "off-topic", "Entwickler-Publisher", "Kritik-Lob-Ideologisch",
                                     "Kritik-Lob-Politisch", "Spielinhalte"]  # "Kritik-Lob-Sonstiges"
    for question in prestudy_annotation_questions:
        answers_1 = annotator_1[question].tolist()
        answers_2 = annotator_2[question].tolist()
        kappa_score = cohen_kappa_score(answers_1, answers_2)
        print(f"Question \"{question}\" - Kappa Score: {kappa_score:.3f}")
        kappa_scores.append(kappa_score)

    average_kappa_score = np.average(kappa_scores)
    print(f"Average kappa score over all questions: {average_kappa_score:.3f}")

    first_annotation_annotator_1 = annotator_1[annotator_1["id"] == 89822457]
    first_annotation_annotator_2 = annotator_2[annotator_2["id"] == 89822457]
    relevant_columns_1 = first_annotation_annotator_1[[*prestudy_annotation_questions]]
    relevant_columns_2 = first_annotation_annotator_2[[*prestudy_annotation_questions]]
    list_1 = relevant_columns_1.values.ravel().tolist()
    list_2 = relevant_columns_2.values.ravel().tolist()
    kappa_score_first = cohen_kappa_score(list_1, list_2)
    print(f"Kappa Score for the very first annotated review: {kappa_score_first:.3f}")


"""
def calculate_cronbach_alpha():
    pre_study_df = pd.read_csv(pathlib.Path(__file__).parent / "exported_data" / "exported_tasks_52140_Vorstudie.csv")
    pre_study_df = pre_study_df.replace({'no': 0, 'yes': 1, 'Nein': 0, 'Ja': 1})
    prestudy_annotation_questions = ["review-bombing", "off-topic", "Entwickler-Publisher", "Kritik-Lob-Ideologisch",
                                     "Kritik-Lob-Politisch", "Spielinhalte"]  # "Kritik-Lob-Sonstiges"

    # convert to this (wide) format:
    # review      is_rb_annotator_1   is_rb_annotator_2 
    # 9934            1                   0
    # 3348            0                   0
    relevant_columns = pre_study_df[["id", "annotator", *prestudy_annotation_questions]]
    wide_df_complete = relevant_columns.pivot(index='id', columns='annotator')
    print(f"Cronbach's Alpha - Wide Df: {pg.cronbach_alpha(data=wide_df_complete)}")
    for question in prestudy_annotation_questions:
        columns = pre_study_df[["id", "annotator", question]]
        wide_df_question = columns.groupby(['id', 'annotator'])[question].first().unstack().reset_index()
        cronbach_alpha_question = pg.cronbach_alpha(data=wide_df_question)
        print(f"Cronbach's Alpha - Question {question}: {cronbach_alpha_question}")
    """


def calculate_krippendorff_alpha(df_file_path):
    """ Interpretation:
    α ≥ 0,800: zuverlässige Übereinstimmung,
    0,800 > α ≥ 0,667: moderat, vorläufige Schlussfolgerungen sind möglich,
    α < 0,667: Daten sind zu verwerfen
    """
    study_df = pd.read_csv(df_file_path)
    relevant_columns = study_df[["id", "annotator", *annotation_questions]]
    relevant_columns = relevant_columns.replace({'Nein': 0, 'Ja': 1})
    # krippendorff alpha is only calculated for reviews that were annotated more than once, so this is not needed:
    # relevant_columns = relevant_columns[relevant_columns.groupby('id').id.transform('count') > 1]

    for question in annotation_questions:
        alpha_value = simpledorff.calculate_krippendorffs_alpha_for_df(relevant_columns, experiment_col='id',
                                                                       annotator_col='annotator', class_col=question)
        print(f"Krippendorff's Alpha for column '{question}': {alpha_value:.4f}")

    """
    ###### Calculating Krippendorff's alpha for file: combined_cleaned_all_projects.csv ######
    Krippendorff's Alpha for column 'is-review-bombing': 0.7095
    Krippendorff's Alpha for column 'is-rating-game-related': 0.4983
    Krippendorff's Alpha for column 'criticism-praise-game-content': 0.6978
    Krippendorff's Alpha for column 'criticism-praise-developer-publisher': 0.5763
    Krippendorff's Alpha for column 'criticism-praise-ideological': 0.2115
    Krippendorff's Alpha for column 'criticism-praise-political': 0.5789
    """


def calculate_final_annotation(review_df: pd.DataFrame, final_annotations: list):
    """
    Applies the majority vote consensus algorithm to the given dataframe in order to calculate one final annotation
    per review for each of the annotation questions. It is expected that the reviews in the given dataframe have
    either 3 annotations or only 1.
    """
    if len(review_df) == 1:
        # for reviews that were annotated only once do nothing, just append them to the output
        final_annotations.append(review_df)
    else:
        # if there are 3 annotators per review take the answer that was given the most often for each annotation column
        if len(review_df) == 2:
            print("WARNING: There are reviews in this project that were annotated only twice! This shouldn't be!")

        last_row = review_df.iloc[-1]  # first, make a copy of the last row and fill it with the aggregated infos
        for annotation_question in annotation_questions:
            # get the answer that was given more often and update the last row copy with it
            most_often_answer = review_df[annotation_question].mode()
            last_row[annotation_question] = most_often_answer[0]

        final_annotations.append(last_row.to_frame().T)


def remove_annotated_duplicates(cleaned_df: pd.DataFrame, rb_incident_name: str):
    # delete all duplicate reviews again (this time punctuation is also removed before comparing which I forgot
    # before uploading ...)
    cleaned_df["review_case_insensitive"] = cleaned_df["review"].astype(str).str.lower()
    cleaned_df["review_case_insensitive"] = cleaned_df['review_case_insensitive'].str.replace(r'[^\w+\s+]', '',
                                                                                              regex=True)
    # create a df without the triple annotated rows first to make it easier to see the actual duplicates
    cleaned_df_no_duplicate_annotator = cleaned_df.drop_duplicates(subset=["id"])
    # cleaned_df_no_duplicates = cleaned_df_no_duplicate_annotator.drop_duplicates(subset=["review_case_insensitive"])
    # print(f"Found {len(cleaned_df_no_duplicate_annotator) - len(cleaned_df_no_duplicates)} rows with duplicate texts")

    # show the duplicate reviews and decide which of them to remove
    duplicates = cleaned_df_no_duplicate_annotator.loc[
                 cleaned_df_no_duplicate_annotator.duplicated(subset=["review_case_insensitive"], keep=False), :]
    # print("\nDuplicate Reviews:")
    # pprint.pprint(duplicates)
    """
    Column IDs to remove:
        Ukraine-Russia: 91992579 (oder 91992491)
        Firewatch: (91871530 oder) 91871603
        Skyrim: 91872684 (oder 91872836)
        AC Unity: 
            91871093 oder 91871260
            91871156 oder 91871282
            91871170 oder 91871271
            91871199 oder 91871331 oder 91871346
            91871297 oder 91871310
            91871308 oder 91871375
    """
    rows_to_remove = []
    if rb_incident_name == "Ukraine-Russia-Support":
        rows_to_remove = [91992579]
    elif rb_incident_name == "Firewatch":
        rows_to_remove = [91871603]
    elif rb_incident_name == "Skyrim - Paid Mods":
        rows_to_remove = [91872684]
    elif rb_incident_name == "Assassin's Creed Unity":
        rows_to_remove = [91871093, 91871156, 91871170, 91871199, 91871331, 91871297, 91871308]
    # Also delete one bad (manually selected) review from the other two incidents that don't have duplicates so all the
    # incidents (apart from AC Unity) have exactly the same number of reviews (to prevent any big biases from the data).
    # One could also remove 7 random reviews from all the other incidents to have the same number as AC Unity but this
    # would be kind of a waste since it has only 7 reviews less and therefore this bias won't be very impactful.
    elif rb_incident_name == "Mortal Kombat 11":
        rows_to_remove = [91871985]
    elif rb_incident_name == "Borderlands - Epic Store":
        rows_to_remove = [91872249]

    # remove the rows with the specified IDs from the dataframe
    cleaned_df = cleaned_df[~cleaned_df["id"].isin(rows_to_remove)]

    cleaned_df = cleaned_df.drop(columns=["review_case_insensitive"], axis=1)  # remove the newly added column again
    return cleaned_df.reset_index(drop=True)


def calculate_correlations(dataframe: pd.DataFrame, column_1, column_2, plot_correlation=False):
    correlation_df = dataframe[[column_1, column_2]]
    correlation_df = correlation_df.replace({'Nein': 0, 'Ja': 1, "Empfohlen": 2, "Nicht empfohlen": 3, "Steam": 4,
                                             "Metacritic": 5})
    corr_matrix = correlation_df.corr(method='pearson')
    print(f"\nCorrelations (Pearson) between columns \"{column_1}\" and \"{column_2}\":")
    pprint.pprint(corr_matrix)

    if plot_correlation:
        # plot the correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        # vmin and vmax control the range of the colormap
        sns.heatmap(corr_matrix, cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax)
        plt.title(f"Correlations between \"{column_1}\" and \"{column_2}\"")
        plt.tight_layout()
        plt.show()

    # contingency table and chi_squared test are better suited for correlation for binary (i.e. nominal) columns
    contingency_table = pd.crosstab(dataframe[column_1], dataframe[column_2])
    print("\nContingency Table:")
    pprint.pprint(contingency_table)
    stat, p_value, dof, expected = chi2_contingency(contingency_table)
    if p_value < 0.05:
        print(f" => Significant Chi^2 correlation (p-value: {p_value:.5f}) between \"{column_1}\" and \"{column_2}\"")


def calculate_overlap_correlation_infos(annotated_dataframe, review_bombing_incident):
    print("\n#################### Statistical Relationships: #########################")
    # calculate correlations between the question columns
    question_combinations = list(itertools.combinations(annotation_questions, 2))
    for question_1, question_2 in question_combinations:
        calculate_correlations(annotated_dataframe, column_1=question_1, column_2=question_2)

    ############################################################################

    # split all reviews into the ones that are seen as review bombing and the ones that aren't
    review_bombing_reviews = annotated_dataframe[(annotated_dataframe["is-review-bombing"] == "Ja")]
    not_review_bombing_reviews = annotated_dataframe[(annotated_dataframe["is-review-bombing"] == "Nein")]
    # also split based on review source
    steam_reviews = annotated_dataframe[(annotated_dataframe["source"] == "Steam")]
    metacritic_reviews = annotated_dataframe[(annotated_dataframe["source"] == "Metacritic")]

    # calculate overlap between Review Bombing - Reviews and the other questions
    for question in annotation_questions:
        if question != "is-review-bombing":
            overlap_df = annotated_dataframe[(annotated_dataframe["is-review-bombing"] == annotated_dataframe[question])]
            print(f"\nOverlap: \"Is Review Bombing\" and \"{question}\": {len(overlap_df)} / {len(annotated_dataframe)}")
            # print(overlap_df['review'].head(5))

            overlap_df_yes = review_bombing_reviews[review_bombing_reviews[question] == "Ja"]
            print(
                f"Number of Review Bombing - Reviews where \"{question}\" == \"Yes\": {len(overlap_df_yes)} / {len(review_bombing_reviews)}")
            overlap_df_no = review_bombing_reviews[review_bombing_reviews[question] == "Nein"]
            print(
                f"Number of Review Bombing - Reviews where \"{question}\" == \"No\": {len(overlap_df_no)} / {len(review_bombing_reviews)}")

    # check if participants answered the two game related questions actually separately from another or not
    overlap_game_related = annotated_dataframe[
        (annotated_dataframe["is-rating-game-related"] == annotated_dataframe["criticism-praise-game-content"])]
    print(f"\nOverlap: \"Is Rating Game Related\" and \"Criticism - Game\": {len(overlap_game_related)} /"
          f" {len(annotated_dataframe)}")

    ############################################################################
    # filter out rows with only one annotation and split into grouped dataframes
    annotated_agreement_df = annotated_dataframe[annotated_dataframe.groupby('id').id.transform('count') > 1]
    rb_reviews_agreement = annotated_agreement_df[(annotated_agreement_df["is-review-bombing"] == "Ja")]
    not_rb_reviews_agreement = annotated_agreement_df[(annotated_agreement_df["is-review-bombing"] == "Nein")]
    steam_reviews_agreement = annotated_agreement_df[(annotated_agreement_df["source"] == "Steam")]
    metacritic_reviews_agreement = annotated_agreement_df[(annotated_agreement_df["source"] == "Metacritic")]

    # check annotator agreement between reviews marked as "review bombing" and reviews that are not considered such
    avg_agreement_rb_yes = round(np.average(rb_reviews_agreement["agreement"]), 2)
    print(f"\nAverage annotator agreement for reviews marked as 'review-bombing': {avg_agreement_rb_yes:.2f} %")
    avg_agreement_rb_no = round(np.average(not_rb_reviews_agreement["agreement"]), 2)
    print(f"Average annotator agreement for reviews not marked as 'review-bombing': {avg_agreement_rb_no:.2f} %")

    # check annotator agreement between review source (i.e. higher agreement on steam or metacritic reviews?)
    avg_agreement_steam = round(np.average(steam_reviews_agreement["agreement"]), 2)
    print(f"Average annotator agreement for Steam reviews: {avg_agreement_steam:.2f} %")
    avg_agreement_metacritic = round(np.average(metacritic_reviews_agreement["agreement"]), 2)
    print(f"Average annotator agreement for Metacritic reviews: {avg_agreement_metacritic:.2f} %\n")

    # calculate correlations between agreement and the questions (mainly is-review-bombing)
    calculate_correlations(annotated_agreement_df, "agreement", "is-review-bombing", plot_correlation=False)

    ############################################################################
    # calculate overlap / correlations between combined rating (i.e. positive/negative) and the questions (mainly
    # is-review-bombing); only makes sense for this one incident as the others are always 100% positive or negative
    if review_bombing_incident == "Ukraine-Russia-Support":
        print("\n######################################### Combined Rating:")
        overlap_rb_rating_negative = review_bombing_reviews[
            review_bombing_reviews["combined_rating"] == "Nicht empfohlen"]
        print(f"\nUkraine-Russia-RB Overlap between \"Combined_Rating\" == 'Nicht empfohlen' and "
              f"\"Is-Review-Bombing\" == 'Ja': {len(overlap_rb_rating_negative)} / {len(review_bombing_reviews)}")
        overlap_rb_rating_positive = review_bombing_reviews[review_bombing_reviews["combined_rating"] == "Empfohlen"]
        print(f"\nUkraine-Russia-RB Overlap between \"Combined_Rating\" == 'Empfohlen' and "
              f"\"Is-Review-Bombing\" == 'Ja': {len(overlap_rb_rating_positive)} / {len(review_bombing_reviews)}")

        calculate_correlations(annotated_dataframe, "is-review-bombing", "combined_rating")

    ############################################################################
    for question in annotation_questions:
        # calculate correlations between source (i.e. Steam / Metacritic) and the questions (mainly is-review-bombing)
        print("\n######################################### Source:")
        overlap_df_steam = steam_reviews[steam_reviews[question] == "Ja"]
        print(
            f"Number of Steam - Reviews where \"{question}\" == \"Yes\": {len(overlap_df_steam)} / {len(steam_reviews)}")
        overlap_df_metacritic = metacritic_reviews[metacritic_reviews[question] == "Ja"]
        print(
            f"Number of Metacritic - Reviews where \"{question}\" == \"Yes\": {len(overlap_df_metacritic)} / {len(metacritic_reviews)}")
        calculate_correlations(annotated_dataframe, "source", question)

        # correlate the review length (i.e. the amount of tokens denoted by "input_ids" column) with the questions
        print("\n######################################### Review Length:")
        calculate_correlations(annotated_dataframe, "review_length", question)
        # TODO calculate correlations between other text features (e.g. sentiment) and the questions ??

    print("\n#################### Statistical Relationships - Analysis finished! #########################\n")


def cleanup_initial_dataframe(exported_dataframe_path: pathlib.Path):
    annotated_df = pd.read_csv(exported_dataframe_path)
    # get the name of the review bombing incident that this data file is for by checking the project ID in the file name
    rb_incident = ""
    for project_id in project_rb_incident_mapping:
        if str(project_id) in exported_dataframe_path.stem:
            rb_incident = project_rb_incident_mapping[project_id]
    if rb_incident == "":
        return None, rb_incident

    # cleanup first: remove columns that aren't needed and reorder the remaining columns
    columns_to_drop = [0, 2, 4, 5, 7, 13, 14, 15, 17, 22, 25, 28, 31, 33, 35]
    cleaned_df = annotated_df.drop(annotated_df.columns[columns_to_drop], axis=1)

    # reorder and rename
    cleaned_df = cleaned_df.iloc[:, [9, 8, 15, 18, 2, 0, 1, 13, 3, 20, 12, 11, 5, 4, 6, 7, 16, 17, 19, 14, 10]]
    cleaned_df = cleaned_df.rename(
        columns={"created_at": "annotation_created_at", "updated_at": "annotation_updated_at",
                 "is-game-related": "is-rating-game-related"})
    # add a new column with the review length based on the number of tokens (not characters) in the review text
    cleaned_df["review_length"] = cleaned_df["input_ids"].str.split(" ").str.len() - 2  # don't ask ...
    # also add the label studio project name as a new column
    cleaned_df.insert(0, "project", [rb_incident] * len(cleaned_df))

    cleaned_df = remove_annotated_duplicates(cleaned_df, rb_incident)
    return cleaned_df, rb_incident


def parse_annotated_csv_data(annotated_data_file: pathlib.Path):
    cleaned_df, rb_incident_name = cleanup_initial_dataframe(exported_dataframe_path=annotated_data_file)
    if rb_incident_name == "":
        print("Could not map file to any review bombing incident!")
        return

    print("\n################################################################")
    print(f"Analyzing annotated data for review bombing incident \"{rb_incident_name}\":\n")

    # consider all lead time entries over 180 seconds (3 min) as outliers and don't include them in the calculations
    lead_time_upper_limit = 180
    lead_times = cleaned_df[cleaned_df["lead_time"] <= lead_time_upper_limit]["lead_time"]
    avg_lead_time_overall = round(np.average(lead_times), 2)
    median_lead_time_overall = round(np.median(lead_times), 2)
    print(f"Average time needed per review for the entire project: {avg_lead_time_overall:.2f} s /"
          f" {avg_lead_time_overall / 60:.2f} min\nMedian time: {median_lead_time_overall:.2f} s /"
          f" {median_lead_time_overall / 60:.2f}")

    # filter out all rows that appear only once in the dataframe before calculating the agreement as these reviews
    # obviously have 100% agreement (and would bias the result)
    agreement_df = cleaned_df[cleaned_df.groupby('id').id.transform('count') > 1]
    avg_agreement_overall = round(np.average(agreement_df["agreement"]), 2)
    print(f"Average annotator agreement for the entire project: {avg_agreement_overall:.2f} %")
    highest_agreement = round(np.max(agreement_df["agreement"]), 2)
    print(f"Highest annotator agreement for this project: {highest_agreement:.2f} %")
    lowest_agreement = round(np.min(agreement_df["agreement"]), 2)
    print(f"Lowest annotator agreement for this project: {lowest_agreement:.2f} %\n")

    calculate_overlap_correlation_infos(cleaned_df, rb_incident_name)

    ############################################################################
    # calculate the counts for the answers for each annotation question and print the aggregated information
    aggregated_question_info = list()
    aggregated_question_annotator_info = list()

    def aggregate_per_annotator(annotator_df: pd.DataFrame, annotation_question):
        annotator_id = annotator_df.reset_index(drop=True).at[0, "annotator"]
        # print(f"\n######### Analyzing annotator with id {annotator_id}: #########")
        column_info = annotator_df[annotation_question].value_counts()
        # print(f"Answers from this annotator for:\n {column_info.to_frame()}")

        avg_lead_time = round(np.average(annotator_df[annotator_df["lead_time"] <= lead_time_upper_limit]["lead_time"]), 2)
        median_lead_time = round(np.median(annotator_df[annotator_df["lead_time"] <= lead_time_upper_limit][
                                           "lead_time"]), 2)
        # print(f"Average time needed for annotating each review: {avg_lead_time:.2f}s / {avg_lead_time / 60:.2f}min")
        avg_agreement = round(np.average(annotator_df["agreement"]), 2)

        annotator_dict = {
            "project": rb_incident_name,
            "annotation_question": annotation_question,
            "num_reviews_yes": column_info.get("Ja", 0),
            "num_reviews_no": column_info.get("Nein", 0),
            "annotator_id": annotator_id,
            "num_annotated_reviews": len(annotator_df),
            "median_lead_time_annotator_in_s": median_lead_time,
            "avg_lead_time_annotator_in_s": avg_lead_time,
            "avg_agreement_annotator": avg_agreement,  # slightly biased for annotators that didn't have a 2nd annotator
        }
        aggregated_question_annotator_info.append(annotator_dict)

    for question in annotation_questions:
        print("#################################")
        question_info = cleaned_df[question].value_counts()
        # print(f"Answers to question \"{question}\":\n{question_info.to_frame()}")

        cleaned_df.groupby(["annotator"], group_keys=False).apply(
            lambda x: aggregate_per_annotator(x, question)).reset_index(drop=True)

        question_dict = {
            "project": rb_incident_name,
            "annotation_question": question,
            "num_reviews_yes": question_info.get("Ja", 0),
            "num_reviews_no": question_info.get("Nein", 0),
            "median_lead_time_project_in_s": median_lead_time_overall,
            "avg_lead_time_project_in_s": avg_lead_time_overall,
            "avg_agreement_project": avg_agreement_overall,
            "highest_agreement_project": highest_agreement,
            "lowest_agreement_project": lowest_agreement,
        }
        aggregated_question_info.append(question_dict)
        # print("#################################\n")

    # convert aggregated infos to dataframes
    aggregated_question_df = pd.DataFrame(aggregated_question_info)
    aggregated_annotator_question_df = pd.DataFrame(aggregated_question_annotator_info)
    aggregated_annotator_df = aggregated_annotator_question_df.sort_values(by=['annotator_id'])

    ############################################################################

    # apply majority voting to get one final annotation for each review
    final_annotations_list = []
    cleaned_df.groupby(["id"], group_keys=False).apply(lambda y: calculate_final_annotation(y, final_annotations_list))
    final_annotations_df = pd.concat(final_annotations_list).reset_index(drop=True)

    # add a new column to the dataframe that indicates whether this review has been annotated once or by 3 annotators
    number_of_annotators_per_review = cleaned_df.groupby('id').id.count()
    final_annotations_df.insert(7, "num_annotators", number_of_annotators_per_review.reset_index(drop=True))

    # remove all annotator-specific columns
    final_annotations_df = final_annotations_df.drop(columns=["annotator", "lead_time", "annotation_created_at",
                                                              "annotation_updated_at"])
    final_annotations_df = final_annotations_df.rename(columns={"agreement": "annotation_certainty"})

    ############################################################################
    # save all relevant dataframes for further use
    cleaned_df.to_csv(OUTPUT_FOLDER / f"cleaned_project_{rb_incident_name}.csv", index=False)
    aggregated_question_df.to_csv(OUTPUT_FOLDER / f"aggregated_question_df_{rb_incident_name}.csv", index=False)
    aggregated_annotator_df.to_csv(OUTPUT_FOLDER / f"aggregated_annotator_df"
                                                   f"_{rb_incident_name}.csv", index=False)
    final_annotations_df.to_csv(OUTPUT_FOLDER / f"final_annotation_project_{rb_incident_name}.csv", index=False)
    print("################################################################\n")


def analyze_aggregated_annotation_information():
    # analyze the annotation infos aggregated over all incidents
    combined_annotator_df = pd.DataFrame()
    combined_question_df = pd.DataFrame()
    combined_cleaned_df = pd.DataFrame()
    combined_final_df = pd.DataFrame()

    for rb_incident in project_rb_incident_mapping.values():
        relevant_files = [parsed_file for parsed_file in OUTPUT_FOLDER.glob(f"*{rb_incident}.csv")]
        for file_path in relevant_files:
            df = pd.read_csv(file_path)
            if "aggregated_annotator" in file_path.stem:
                combined_annotator_df = pd.concat([combined_annotator_df, df], ignore_index=True)
            elif "aggregated_question" in file_path.stem:
                combined_question_df = pd.concat([combined_question_df, df], ignore_index=True)
            elif "final_annotation" in file_path.stem:
                combined_final_df = pd.concat([combined_final_df, df], ignore_index=True)
            elif "cleaned_project" in file_path.stem:
                # cleaned_project is basically the final_annotation_project file but with all annotators instead of
                # the combined annotation (and some more annotator-specific columns)
                combined_cleaned_df = pd.concat([combined_cleaned_df, df], ignore_index=True)

    ############################################################################
    avg_lead_time = round(np.average(combined_annotator_df["avg_lead_time_annotator_in_s"]), 2)
    median_lead_time = round(np.average(combined_annotator_df["median_lead_time_annotator_in_s"]), 2)
    print(f"Average time needed per review over all projects: {avg_lead_time:.2f} s / {avg_lead_time / 60:.2f} "
          f"min\nMedian time: {median_lead_time:.2f} s / {median_lead_time / 60:.2f}\n")
    # also show the lead time per annotator aggregated over all projects
    avg_annotator_lead_times = combined_annotator_df.groupby("annotator_id")["avg_lead_time_annotator_in_s"].agg(np.average)
    # median_annotator_lead_times = combined_annotator_df.groupby("annotator_id")["median_lead_time_annotator_in_s"].agg(np.average)

    # plot the annotator lead times
    sns.set_style("darkgrid")
    ax = sns.barplot(x=avg_annotator_lead_times.index, y=avg_annotator_lead_times.values)
    ax.set(xlabel="Annotator ID", ylabel="Average time in seconds")  # , ylim=(0, 200)
    ax.set_title("Average annotation time for each annotator")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "average_lead_time_per_annotator.svg", format="svg")

    ############################################################################
    # show annotator agreement aggregated over all projects
    avg_agreement = np.average(combined_question_df["avg_agreement_project"])
    print(f"Average annotator agreement over all projects: {avg_agreement:.2f} %")

    agreement_df = combined_cleaned_df[combined_cleaned_df.groupby('id').id.transform('count') > 1]
    # check annotator agreement between reviews marked as "review bombing" and reviews that are not considered such
    avg_agreement_rb_yes = np.average(agreement_df[agreement_df["is-review-bombing"] == "Ja"]["agreement"])
    print(f"Average annotator agreement for reviews marked as 'review-bombing': {avg_agreement_rb_yes:.2f} %")
    avg_agreement_rb_no = np.average(agreement_df[agreement_df["is-review-bombing"] == "Nein"]["agreement"])
    print(f"Average annotator agreement for reviews marked as 'not review-bombing': {avg_agreement_rb_no:.2f} %")
    # check annotator agreement between review source (i.e. higher agreement on steam or metacritic reviews)
    avg_agreement_steam = np.average(agreement_df[agreement_df["source"] == "Steam"]["agreement"])
    print(f"Average annotator agreement for Steam reviews: {avg_agreement_steam:.2f} %")
    avg_agreement_metacritic = np.average(agreement_df[agreement_df["source"] == "Metacritic"]["agreement"])
    print(f"Average annotator agreement for Metacritic reviews: {avg_agreement_metacritic:.2f} %\n")

    ############################################################################
    # check question answers grouped by source (Steam / Metacritic) and review bombing type (negative / positive / both)
    def show_question_answers_overall(dataframe: pd.DataFrame):
        group_name = dataframe.name
        for question in annotation_questions:
            aggr_question_info = dataframe[question].value_counts()
            # pprint.pprint(aggr_question_info)
            info_dict = {
                "group": group_name,
                "annotation_question": question,
                "Yes": aggr_question_info.get("Ja", 0),  # Applies to review - yes
                "No": aggr_question_info.get("Nein", 0),  # Applies to review - no
            }
            plot_info.append(info_dict)

    plot_info = list()
    # print(f"\nQuestion answers over all projects grouped by \"combined_rating\":")
    combined_final_df.groupby("combined_rating").apply(lambda frame: show_question_answers_overall(frame))
    rating_plot_df_long = pd.melt(pd.DataFrame(plot_info), id_vars=['group', 'annotation_question'])
    # print(f"\nQuestion answers over all projects grouped by \"source\":")
    combined_final_df.groupby("source").apply(lambda frame: show_question_answers_overall(frame))
    source_plot_df = pd.DataFrame(plot_info)
    source_plot_df_long = pd.melt(source_plot_df, id_vars=['group', 'annotation_question'])
    combined_plot_df_long = pd.concat([rating_plot_df_long, source_plot_df_long])

    # plot both groups
    plot = sns.catplot(data=combined_plot_df_long, x="annotation_question", y="value", col="group", hue="variable",
                       kind="bar", col_wrap=2, aspect=.8, dodge=True)
    plot.set_axis_labels("", "Number of reviews")
    plot.fig.suptitle('Annotation question answers over all projects: Reviews grouped by - ', fontsize=10)
    plot.set_titles("{col_name}")   # - \"{col_var}\"")
    # plot.fig.subplots_adjust(top=0.8)  # adjust space between plot and title
    plot.tick_params(axis='x', rotation=75)
    plot.tight_layout()
    plt.savefig(PLOT_FOLDER / "grouped_question_answers_rating_source.png", format="png")
    plt.savefig(PLOT_FOLDER / "grouped_question_answers_rating_source.svg", format="svg")

    ############################################################################
    # calculate correlations between projects
    # first, split into separate dataframes for each project
    project_df_list = [group_df for group_name, group_df in combined_final_df.groupby("project")]

    for project_1, project_2 in list(itertools.combinations(project_df_list, 2)):
        # very important to reset index before calculating the correlation !!
        df_1 = project_1.reset_index(drop=True)
        df_2 = project_2.reset_index(drop=True)
        print(f"\n###################### Correlations between projects \"{df_1.at[0, 'project']}\" and "
              f"\"{df_2.at[0, 'project']}\":")
        df_1 = df_1[[*annotation_questions]].replace({'Nein': 0, 'Ja': 1})
        df_2 = df_2[[*annotation_questions]].replace({'Nein': 0, 'Ja': 1})
        df_combined = df_1.join(df_2, rsuffix='_2')
        corr_matrix = df_combined.corr(method="pearson")
        # 'remove' the cells that don't indicate a significant correlation
        significant_correlations = corr_matrix.where((corr_matrix < 0.05) & (corr_matrix > -0.05), '')
        pprint.pprint(significant_correlations)
        # significant_corr = corr_matrix.apply(lambda row: row[(row < 0.05) & (row > -0.05)].index.tolist(), axis=1)

    # compare some differences between the projects in general
    #  -> notable differences between ac unity vs. ukraine-russia vs the rest: fewer reviews marked as rb
    #  for projects ac unity and ukraine-russia than for the others (maybe because those are the only two that
    #  are / include positive review bombing?)
    project_info_list = list()
    for project in project_df_list:
        project_name = project.reset_index(drop=True).at[0, "project"]
        print(f"\n######################### Project \"{project_name}\":")
        num_review_bombing_reviews = project["is-review-bombing"].value_counts().get("Ja", 0)
        print(f"Number of reviews marked as 'Review Bombing': {num_review_bombing_reviews}")

        num_off_topic_reviews = project["is-rating-game-related"].value_counts().get("Nein", 0)
        print(f"Number of reviews marked as 'Off Topic': {num_off_topic_reviews}")

        project_info_dict = {
            "Project": project_name,
            "Review Bombing Reviews": (num_review_bombing_reviews / len(project)) * 100,
            "Off Topic Reviews": (num_off_topic_reviews / len(project)) * 100,
        }
        project_info_list.append(project_info_dict)

    project_info_df = pd.DataFrame(project_info_list)
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    sns.barplot(data=project_info_df, x="Project", y="Review Bombing Reviews", hue="Project", ax=ax1)
    sns.barplot(data=project_info_df, x="Project", y="Off Topic Reviews", hue="Project", ax=ax2)
    fig.suptitle("Target Label Annotation per Project", fontsize=12)
    ax1.set_ylabel("Percentage of Review Bombing - Reviews")
    ax2.set_ylabel("Percentage of Off Topic - Reviews")
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax1.set(xticklabels=[])
    ax2.set(xticklabels=[])
    # ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # ax2.set_ylim([60, 100])

    ax1.get_legend().remove()
    sns.move_legend(
        ax2, "upper center",
        bbox_to_anchor=(1, 1), ncol=1, title=None, frameon=False, borderaxespad=0.,
    )
    """
    project_info_df_long = pd.melt(project_info_df, id_vars=['Project'])
    project_info_df_long = project_info_df_long.rename(columns={"variable": "Percentage"})
    ax = sns.catplot(data=project_info_df_long, x="Project", y="value", hue="Percentage",
                     kind="bar", dodge=True, legend_out=False)
    ax.set_axis_labels("", "Percentage of all Reviews")
    ax.fig.suptitle('Target Label Annotation per Project')
    ax.tick_params(axis='x', rotation=65)

    plt.tight_layout()
    plt.savefig(PLOT_FOLDER / "grouped_target_project_features.svg", format="svg")

    # plt.show()
    ############################################################################

    combined_annotator_df.to_csv(OUTPUT_FOLDER / "combined_annotators_all_projects.csv", index=False)
    combined_question_df.to_csv(OUTPUT_FOLDER / "combined_questions_all_projects.csv", index=False)
    combined_cleaned_df.to_csv(OUTPUT_FOLDER / "combined_cleaned_all_projects.csv", index=False)
    combined_final_df.to_csv(OUTPUT_FOLDER / "combined_final_annotation_all_projects.csv", index=False)
    print("################################################################\n")


def plot_annotator_agreement():
    all_incidents = pd.read_csv(OUTPUT_FOLDER / "combined_final_annotation_all_projects.csv")
    project_agreement = list()

    def calculate_agreement_per_project(group_df: pd.DataFrame):
        incident_name = group_df.name
        group_df = group_df.reset_index(drop=True)
        test = group_df["annotation_certainty"]
        avg_agreement_overall = round(np.average(test), 2)
        rb_s = group_df[group_df["is-review-bombing"] == "Ja"]["annotation_certainty"]
        not_rb_s = group_df[group_df["is-review-bombing"] == "Nein"]["annotation_certainty"]
        avg_agreement_rb_reviews = round(np.average(rb_s), 2)
        avg_agreement_not_rb_reviews = round(np.average(not_rb_s), 2)

        project_agreement_dict = {
            "Project": incident_name,
            "Avg Agreement Overall": avg_agreement_overall,
            "Avg Agreement Is-Review-Bombing": avg_agreement_rb_reviews,
            "Avg Agreement Is-Not-Review-Bombing": avg_agreement_not_rb_reviews,
        }
        project_agreement.append(project_agreement_dict)

    # remove reviews that were annotated only once first to not bias the agreement results
    filtered_incidents = all_incidents[all_incidents["num_annotators"] > 1]
    filtered_incidents.groupby("project").apply(calculate_agreement_per_project)

    project_agreement_df = pd.DataFrame(project_agreement)
    project_agreement_df_long = pd.melt(project_agreement_df, id_vars=['Project'])
    project_agreement_df_long = project_agreement_df_long.rename(columns={"variable": "Type"})

    # plt.rcParams["figure.figsize"] = (8, 12)
    plot = sns.catplot(data=project_agreement_df_long, x="Project", y="value", hue="Type",
                       kind="bar", dodge=True, legend_out=False, aspect=11.7/8.27)
    plot.set_axis_labels("", "Agreement in %")
    plot.fig.suptitle('Annotator agreement for all projects')
    plot.tick_params(axis='x', rotation=65)
    plot.set(ylim=(60, 100))
    plot.tight_layout()
    plt.savefig(PLOT_FOLDER / "annotator_agreement_per_project.svg", format="svg")
    # plt.show()


"""
def parse_annotated_json_data(annotated_data_file):
    annotated_content = json.load(open(annotated_data_file, "r", encoding="utf8"))
    # for annotation in annotated_content:
    print("Not implemented yet :(")

    # Retrieve two annotations in the Label Studio JSON format
    # r1 = annotation_1["result"][0]["value"]["choices"][0][0]
    # r2 = annotation_2["result"][0]["value"]["choices"][0][0]
"""


if __name__ == "__main__":
    pd.options.display.width = 0
    # pd.set_option('future.no_silent_downcasting', True)  # disable some warnings

    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir(parents=True)
    if not PLOT_FOLDER.is_dir():
        PLOT_FOLDER.mkdir()

    run_parsing_code = False
    single_input_file = False

    if run_parsing_code:
        if single_input_file:
            input_file = DATA_FOLDER / "exported_tasks_project_52575.csv"
            parse_annotated_csv_data(input_file)
        else:
            for file in pathlib.Path.iterdir(DATA_FOLDER):
                print(f"\n################## Parsing file {file.name} ##################\n")
                if file.suffix == ".json":
                    # parse_annotated_json_data(DATA_FOLDER / file)
                    continue  # for now, only parse csv files
                elif file.suffix == ".csv":
                    parse_annotated_csv_data(DATA_FOLDER / file)
                else:
                    raise ValueError("The file is neither json nor csv.")

    analyze_aggregated_information = True
    if analyze_aggregated_information:
        analyze_aggregated_annotation_information()
        plot_annotator_agreement()

    calculate_krippendorff_metric = False
    if calculate_krippendorff_metric:
        for file in pathlib.Path(OUTPUT_FOLDER).glob("cleaned_project*.csv"):
            print(f"\n###### Calculating Krippendorff's alpha for file: {file.name} ######\n")
            calculate_krippendorff_alpha(OUTPUT_FOLDER / file)
