#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def analyze_correlations_with_annotations(df: pd.DataFrame):
    # analyze whether people with pre-knowledge about review bombing or video game reviews had higher annotator
    # agreement / lower annotation times; maybe even correlation with the number of reviews markes as review bombing ?
    annotators_df = pd.read_csv(pathlib.Path(__file__).parent / "parsed_data" / "combined_annotators_all_projects.csv")
    annotators_df_rb_rows = annotators_df[annotators_df["annotation_question"] == "is-review-bombing"]

    # also add column "Wie häufig spielen Sie in der Regel Videospiele?" ?
    experience_columns = ['Kannten Sie das Phänomen "Review Bombing" bereits vorher?', "Wie häufig lesen Sie auf "
                          "Plattformen, wie zum Beispiel Steam oder Metacritic, Nutzerreviews zu Videospielen ?"]
    annotation_columns = ["num_reviews_yes", "num_reviews_no", "avg_lead_time_annotator_in_s",
                          "avg_agreement_annotator"]
    df_1 = df[["annotator_id", *experience_columns]]
    df_2 = annotators_df_rb_rows[["annotator_id", *annotation_columns]]

    # calculate the average values for each annotator over all projects and questions
    df_2_cleaned = df_2.groupby("annotator_id", as_index=False).agg(np.average)
    merged_df = pd.merge(df_1, df_2_cleaned, on='annotator_id', how='inner')

    # plot the average values for the experience questions
    for x_column in experience_columns:
        merged_df_long = pd.melt(merged_df, id_vars=[x_column], value_vars=["num_reviews_yes", "num_reviews_no"])
        plot = sns.catplot(data=merged_df_long, x=x_column, y="value", hue="variable", kind="bar", aspect=.8, width=.7,
                           dodge=True)
        plot.set_axis_labels(x_column, "Average number of Reviews")
        plot.fig.suptitle(f'Number of Reviews marked as (Not) Review Bombing for question')
        plot.tight_layout()

        merged_df_long = pd.melt(merged_df, id_vars=[x_column], value_vars=["avg_agreement_annotator"])
        plot2 = sns.catplot(data=merged_df_long, x=x_column, y="value", kind="bar", aspect=.8, width=0.5)
        plot2.set_axis_labels(x_column, "Average annotator agreement in %")
        plot2.fig.suptitle(f'Annotator agreement for question \n\"{x_column}\"')
        plot2.tight_layout()

        merged_df_long = pd.melt(merged_df, id_vars=[x_column], value_vars=["avg_lead_time_annotator_in_s"])
        plot3 = sns.catplot(data=merged_df_long, x=x_column, y="value", kind="bar", aspect=.8, width=0.5)
        plot3.set_axis_labels(x_column, "Average annotator lead time in s")
        plot3.fig.suptitle(f'Annotator Lead Time for question \n\"{x_column}\"')
        plot3.tight_layout()
        plt.show()

    # calculate the correlations between the average values and the experience questions
    merged_df_numeric = merged_df.replace({'Nein': 0, 'Ja': 1, "Sehr häufig": 2, "Regelmäßig": 3, "Eher selten": 4,
                                          "Nie oder fast nie": 5})
    # corr_matrix = merged_df_numeric.corr(method="pearson")
    # pprint.pprint(corr_matrix)
    for column_1 in experience_columns:
        for column_2 in annotation_columns:
            correlation_df = merged_df_numeric[[column_1, column_2]]
            corr_matrix = correlation_df.corr(method='pearson')
            print(f"\nCorrelation {corr_matrix.iat[1, 0]:.4f} between columns \"{column_1}\" and \"{column_2}\"")


def analyze_demographics(df: pd.DataFrame):
    demographics_columns = df[["Geschlecht", "Alter", "Beruf (als Student bitte Studienfach angeben)"]]
    pprint.pprint(demographics_columns.describe(include='all').T)
    pprint.pprint(demographics_columns["Beruf (als Student bitte Studienfach angeben)"].value_counts())


def analyze_prior_experience(df: pd.DataFrame):
    experience_columns = df[["Wie häufig spielen Sie in der Regel Videospiele?",
                             "Wie häufig lesen Sie auf Plattformen, wie zum Beispiel Steam oder Metacritic, "
                             "Nutzerreviews zu Videospielen ?",
                             'Kannten Sie das Phänomen "Review Bombing" bereits vorher?',
                             "Falls ja, in welchem Kontext war Ihnen Review Bombing bereits bekannt?"]]
    pprint.pprint(experience_columns["Wie häufig spielen Sie in der Regel Videospiele?"].value_counts())
    print("\n")
    pprint.pprint(experience_columns["Wie häufig lesen Sie auf Plattformen, wie zum Beispiel Steam oder Metacritic, "
                                     "Nutzerreviews zu Videospielen ?"].value_counts())
    print("\n")
    pprint.pprint(experience_columns['Kannten Sie das Phänomen "Review Bombing" bereits vorher?'].value_counts())

    print("\n######## Contingency Tables: ########\n")
    # relationship between amount of playing games and reading game reviews
    contingency_table = pd.crosstab(df["Wie häufig spielen Sie in der Regel Videospiele?"], df[
        "Wie häufig lesen Sie auf Plattformen, wie zum Beispiel Steam oder Metacritic, Nutzerreviews zu Videospielen ?"])
    pprint.pprint(contingency_table)
    print("\n")
    # relationship between amount of playing games and knowing review bombing
    contingency_table = pd.crosstab(df["Wie häufig spielen Sie in der Regel Videospiele?"],
                                    df['Kannten Sie das Phänomen "Review Bombing" bereits vorher?'])
    pprint.pprint(contingency_table)
    print("\n")
    # relationship between reading game reviews and knowing review bombing
    contingency_table = pd.crosstab(df["Wie häufig lesen Sie auf Plattformen, wie zum Beispiel Steam oder Metacritic, Nutzerreviews zu Videospielen ?"],
                                    df['Kannten Sie das Phänomen "Review Bombing" bereits vorher?'])
    pprint.pprint(contingency_table)

    """
    ax = sns.countplot(data=experience_columns, x="Wie häufig spielen Sie in der Regel Videospiele?", width=0.5)
    ax.set(xlabel="", ylabel="Anzahl")
    ax.set_title("Wie häufig werden Videospiele gespielt?")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()

    ax2 = sns.countplot(data=experience_columns, x="Wie häufig lesen Sie auf Plattformen, wie zum Beispiel Steam oder Metacritic, Nutzerreviews zu Videospielen ?", width=0.5)
    ax2.set(xlabel="", ylabel="Anzahl")
    ax2.set_title("Häufigkeit gelesener Videospiel-Reviews")
    ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    """


if __name__ == "__main__":
    pd.options.display.width = 0
    sns.set_style("darkgrid")

    questionnaire_df = pd.read_csv("MA-Pre-Fragebogen.csv")

    # remove participants who dropped out of the study
    questionnaire_df = questionnaire_df.drop([0, 3], axis=0).reset_index(drop=True)
    # also remove columns that are not needed
    questionnaire_df = questionnaire_df.drop(questionnaire_df.columns[[8, 9, 10]], axis=1)
    # add the label studio IDs to the questionnaire so the answers can be combined with the data from the annotators
    label_studio_ids = pd.Series([17568, 15655, 17629, 9365, 9627, 15657, 17750, 17758, 13430, 17811, 17911, 9368,
                                  16195, 18232, 10389, 18236, 18248, 11790, 11478, 9754, 15710, 18440, 11926, 19069,
                                  19146, 19561])
    questionnaire_df["annotator_id"] = label_studio_ids.values

    print("\n################# Analyzing demographics: ##########################\n")
    analyze_demographics(questionnaire_df)
    print("\n################# Analyzing prior experience: ##########################\n")
    analyze_prior_experience(questionnaire_df)
    print("\n################# Analyzing correlations with annotations: ##########################\n")
    analyze_correlations_with_annotations(questionnaire_df)
