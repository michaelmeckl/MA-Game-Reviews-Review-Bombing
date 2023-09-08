import ast
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_colwidth', None)


def some_pandas_examples():
    dict_of_dicts = {"Alter": {"Hans": 24, "Paula": 22, "Peter": 47},
                     "Studiengang": {"Hans": "IW", "Paula": "IW", "Peter": "MI"}}
    df = pd.DataFrame(dict_of_dicts)

    # show datatypes
    print(df.dtypes)

    # change datatype
    # df['Alter'] = df['Alter'].astype(int)

    # get column
    print(df["Alter"])

    # get row slice
    print(df["Hans":"Peter"])

    # get rows based on column(s) with 'loc(rows, columns)'
    print(df.loc["Paula", "Alter":"Studiengang"])
    print(df.loc["Paula"])

    # use indices instead of names with iloc()
    print(df.iloc[0:3, 0])

    # get / set specific cell value
    df.at["Hans", "Alter"] = 35
    print(df.at["Hans", "Alter"])

    # group by aggregator
    print(df.groupby("Studiengang")[["Alter"]].sum())

    # use masks to filter data
    mask = df["Alter"] > 24
    print(mask)
    print(df[mask])

    # iterate over rows & columns (Iterating should generally be avoided in pandas as it is inefficient !)
    # for index, value in df.iterrows():

    # add rows to dataframe by appending a dictionary
    # TODO does not work anymore in pandas v2.0+ since append is deprecated => use pd.concat([]) instead, see below
    df_new = df.append({'Alter': {'Daniel': 27}, 'Studiengang': {'Daniel': 'MI'}}, ignore_index=True)
    # or by converting a list to a pandas-Series
    df_new = df.append(pd.Series([2345, 2, 650], index=df.columns), ignore_index=True)

    # combine two dataframes
    example_array = np.arange(1, 51).reshape(5, 10)  # create matrix with 5 rows and 10 columns
    new_df = pd.DataFrame(example_array)
    df_combined = pd.concat([df, new_df])

    # add column to dataframe
    df['Semester'] = '5. Semester'

    # rename columns
    df = df.rename(columns={'Semester': 'SemesterNew'})

    # remove columns
    df = df.drop(columns=['SemesterNew'], axis=1)

    # remove duplicates
    df_new = df.drop_duplicates(subset=['text'], keep='first')

    # Select columns
    df = df[['Alter', 'Studiengang']]

    # Select only one column and convert to list
    print(list(df['Alter']))

    # get values with counts per column
    df['Alter'].value_counts()


def init_df_from_file(csv_file):
    # check if the file already exists
    if os.path.isfile(csv_file):
        study_data = pd.read_csv(csv_file)
    else:
        # init empty dataframe with the necessary columns instead
        study_data = pd.DataFrame(columns=['timestamp', 'participantID', 'condition'])
    return study_data


def init_df_from_file_alternative(csv_file_path):
    """
    Alternative variant that uses a converter to extract a list from a csv cell
    """
    # check if the file already exists (this time with pathlib)
    if csv_file_path.exists():
        study_data = pd.read_csv(csv_file_path, sep=";")
        # use a converter to convert saved list back to a list (by default it would be a string)
        study_data['list_data'] = study_data['list_data'].apply(ast.literal_eval)
        return study_data


def save_df_to_csv(df, out_file):
    df = df.append({'timestamp': time.time(), 'participantID': 1, 'condition': 'A'}, ignore_index=True)
    df.to_csv(out_file, index=False)


def save_df_to_csv_alternative(df, out_file):
    """
    Alternative variant that saves a list to a csv cell, see 'condition' - field
    """
    df = df.append({'timestamp': time.time(), 'participantID': 1, 'condition': ['A', 'B']}, ignore_index=True)
    df.to_csv(out_file, sep=";", index=False)


def visualize_df(df_results):
    # split & filter data
    df_word_events = df_results[df_results['event_type'] == 'EventTypes.WORD_TYPED']
    words_condition_A_duration = df_word_events[df_word_events['condition'] == 'A']['duration']
    words_condition_B_duration = df_word_events[df_word_events['condition'] == 'B']['duration']

    # show barplot with matplotlib
    plt.title("Mean Reaction times for conditions")
    plt.bar("Condition A", words_condition_A_duration.mean())
    plt.bar("Condition B", words_condition_B_duration.mean())

    # show barplot with seaborn
    ax = plt.axes()
    sns.barplot(data=df_word_events, x="participant_id", y="duration", hue="condition", ax=ax)
    ax.set_xlabel("participant id") \
        .set_ylabel("time in seconds") \
        .set_title("Time needed per word")

    # show prettier barplot with catplot
    plot = sns.catplot(x="participant_id", y="duration", hue="condition", kind="bar", col="condition", aspect=.75,
                       dodge=False, data=df_word_events)
    plot.fig.subplots_adjust(top=0.8)  # add some space between plot and title
    plot.fig.suptitle('Time needed per word')

    # show scatterplot
    ax = plt.axes()
    sns.scatterplot(data=df_word_events, x="duration", y="participant_id", hue="condition", ax=ax)
    ax.set_title("Duration by ID for both conditions")

    # alternatively a catplot can be used here as well:
    # sns.catplot(kind="swarm", ...)

    # show boxplot
    ax = plt.axes()
    sns.boxplot(data=df_word_events, x="condition", y="duration")
    ax.set_title("Duration for condition B and A")
