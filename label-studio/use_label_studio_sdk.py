#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pprint
import pathlib
import pandas as pd
from label_studio_sdk import Client
from label_studio_sdk.project import Project
from label_studio_sdk.users import User, UserRole
from label_studio_sdk.data_manager import Filters, Column, Operator, Type
from dotenv import load_dotenv


def export_tasks(project_obj: Project, export_format: str):
    print(f"Exporting annotated tasks for project {project_obj.id} in format {export_format} ...")
    output_folder = pathlib.Path(__file__).parent / "exported_data"
    if not output_folder.is_dir():
        output_folder.mkdir()

    output_name = f"exported_tasks_project_{project_obj.id}"
    if export_format == "JSON":
        # export only annotated tasks
        result = project_obj.export_tasks(export_type=export_format, download_all_tasks=False)
        df = pd.DataFrame.from_dict(result)
        df.to_json(output_folder / f"{output_name}.json", orient="records")

    elif export_format == "CSV":
        project_obj.export_tasks(export_type=export_format, download_all_tasks=False,
                                 export_location=output_folder / f"{output_name}.csv")
    else:
        print("Unknown export format!")


def export_reviews_from_projects(project_ids: list, export_format="CSV"):
    # export annotated tasks for each project in IDs
    for project_id in project_ids:
        project = ls.get_project(id=project_id)
        export_tasks(project, export_format)


def add_user_to_project(user: User, project_ids: list):
    user.set_role(UserRole.ANNOTATOR)

    for project_id in project_ids:
        # get project and add user to the project
        project = ls.get_project(id=project_id)
        project.add_member(user=user)
        print(f"Added user \"{user.email}\" to project: {project.id}")

        # check that it worked:
        all_project_members = project.get_members()
        print(f"All members of project: {len(all_project_members)}")
        # pprint.pprint(all_project_members)
        print(f"######################################\n")


"""
def get_labeled_tasks(project_obj: Project):
    print(f"\nPrinting labeled tasks for project {project_obj.id}:")
    annotated_reviews = project_obj.get_labeled_tasks(only_ids=False)
    print(f"Found {len(annotated_reviews)} annotated tasks:")
    pprint.pprint(annotated_reviews)
    print("#########################################")
"""


def construct_annotator_filter(annotator_ids):
    all_filter_items = []
    for annotator_id in annotator_ids:
        filter_item = Filters.item(
            Column.annotators,
            Operator.CONTAINS,
            Type.List,
            Filters.value(annotator_id)
        )
        all_filter_items.append(filter_item)

    final_filter = Filters.create(Filters.OR, all_filter_items)
    return final_filter


def get_already_annotated_tasks(project: Project, num_tasks_per_project, use_reviews_from_specific_annotators=True):
    # filter only the reviews that are already annotated exactly once and if wanted only from certain first-annotators
    filter_annotated = Filters.create(Filters.AND, [
            Filters.item(
                Column.total_annotations,
                Operator.EQUAL,
                Type.Number,
                Filters.value(1)
            ),
            # also check if they are already completed, i.e. empty == false/0; if they have exactly one annotation but
            # are not completed yet, they were already assigned to a second annotator (we don't want to assign a third)
            Filters.item(
                Column.completed_at,
                Operator.EMPTY,
                Type.Number,
                Filters.value(0)
            ),
        ],
    )

    if use_reviews_from_specific_annotators:
        annotator_ids = [9627]  # TODO preferably not 17750
        # get all ids from the specified annotators by creating a filter item for each id and combining them with OR
        filter_annotators = construct_annotator_filter(annotator_ids)
        ids_for_annotators = project.get_tasks(filters=filter_annotators, only_ids=True)

        # get all ids that were already annotated exactly once
        annotated_ids = project.get_tasks(filters=filter_annotated, only_ids=True)

        # only take the id from the specified annotators if they are already annotated exactly once
        task_ids = [_id for _id in ids_for_annotators if _id in annotated_ids]
    else:
        task_ids = project.get_tasks(filters=filter_annotated, only_ids=True)

    if num_tasks_per_project > len(task_ids):
        print(f"Not enough tasks could be found! There are {num_tasks_per_project - len(task_ids)} tasks left to "
              f"be assigned!")
        return task_ids
    else:
        return task_ids[:num_tasks_per_project]  # select the first 'num_tasks_per_project' elements


def get_double_annotated_tasks(project: Project, num_tasks_per_project):
    filter_annotated_twice = Filters.create(Filters.AND, [
        Filters.item(
            Column.total_annotations,
            Operator.EQUAL,
            Type.Number,
            Filters.value(2)
        ),
        Filters.item(
            Column.completed_at,
            Operator.EMPTY,
            Type.Number,
            Filters.value(0)
        ),
    ])

    task_ids = project.get_tasks(filter_annotated_twice, only_ids=True)
    if num_tasks_per_project > len(task_ids):
        print(f"WARNING: There are only {len(task_ids)} tasks in this project that were already annotated twice! "
              f"Cannot assign {num_tasks_per_project} tasks! Cancelling assignment ...")
        return []
    else:
        return task_ids[:num_tasks_per_project]


def get_unassigned_tasks(project: Project, num_tasks_per_project):
    # find the next 'num_tasks_per_project' tasks in the project that are not yet assigned to an annotator (i.e. where
    # the annotator column is empty)
    # TODO replace with this later again
    """
    Filters.item(
        Column.annotators,
        Operator.EMPTY,
        Type.Unknown,
        Filters.value(1)
    )
    """
    filter_unassigned = Filters.create(
        Filters.OR,
        [
            Filters.item(
                Column.total_annotations,
                Operator.EQUAL,
                Type.Number,
                Filters.value(0)
            )
        ],
    )
    task_ids = project.get_tasks(filters=filter_unassigned, only_ids=True)
    if num_tasks_per_project > len(task_ids):
        print(f"WARNING: There are only {len(task_ids)} unassigned tasks in this project left! Cannot assign {num_tasks_per_project} tasks!")
        return []

    """
    start_id = task_ids[0]
    end_id = (start_id + num_tasks_per_project) - 1
    print(f"Next unassigned task ID is: {start_id}")

    # Get the next n tasks for this project
    filter_correct_number = Filters.create(
        Filters.OR,
        [
            Filters.item(Column.id, Operator.IN, Type.Number, Filters.value(start_id, end_id)),
        ],
    )
    return project.get_tasks(filters=filter_correct_number, only_ids=True)
    """
    return task_ids[:num_tasks_per_project]


def assign_user_to_tasks(project_ids: list, user: User, num_tasks_per_project: int,
                         assign_to_once_annotated_tasks: bool, assign_to_double_annotated_tasks: bool):
    for project_id in project_ids:
        project = ls.get_project(project_id)
        # get the correct tasks to assign to this user
        if assign_to_once_annotated_tasks:
            print("Assigning to tasks that were already annotated once ...")
            found_tasks = get_already_annotated_tasks(project, num_tasks_per_project)
        elif assign_to_double_annotated_tasks:
            print("Assigning to tasks that were already annotated twice ...")
            found_tasks = get_double_annotated_tasks(project, num_tasks_per_project)
        else:
            print("Assigning to not annotated tasks ...")
            found_tasks = get_unassigned_tasks(project, num_tasks_per_project)

        if not found_tasks:
            print(f'No tasks found for project {project_id}!')
            continue

        project.assign_annotators([user], found_tasks)
        print(f'User {user.username} was assigned to {len(found_tasks)} tasks from id={found_tasks[0]} to id'
              f'={found_tasks[-1]} in project {project_id}')


def get_task_information_for_annotator(project_ids: list, user: User):
    filter_assigned = Filters.create(Filters.OR, [
        Filters.item(
            Column.annotators,
            Operator.CONTAINS,
            Type.List,
            Filters.value(user.id)
        )
    ])

    filter_assigned_and_completed = Filters.create(Filters.AND, [
        Filters.item(
            Column.annotators,
            Operator.CONTAINS,
            Type.List,
            Filters.value(user.id)
        ),
        # check if they are marked as completed, i.e. empty == false (0)
        Filters.item(
            Column.completed_at,
            Operator.EMPTY,
            Type.Number,
            Filters.value(0)
        ),
    ])

    for project_id in project_ids:
        project = ls.get_project(project_id)
        assigned_tasks = project.get_tasks(filters=filter_assigned, only_ids=True)
        completed_tasks = project.get_tasks(filters=filter_assigned_and_completed, only_ids=True)

        print(f"\n######################################")
        print(f"Found {len(assigned_tasks)} tasks ({len(completed_tasks)} completed) for user {user.username} in project {project.id}")
        print(f"######################################")


def assign_reviews(is_first_assignment, assign_to_once_annotated_tasks, assign_to_double_annotated_tasks):
    print("################################\n")
    # find the correct user in the organization, add user as Annotator to all projects and assign the tasks correctly
    all_users = ls.get_users()
    found_users = [u for u in all_users if u.email.lower() == user_email.lower()]
    if len(found_users) == 0:
        print(f"Couldn't find user with email \"{user_email}\" :(")
    else:
        found_user = found_users[0]
        print(f"Found user: {found_user}")

        if is_first_assignment:
            # only call this once per user!
            add_user_to_project(found_user, project_ids=study_project_ids)

        # assign tasks
        assign_user_to_tasks(study_project_ids, found_user, number_tasks_per_project,
                             assign_to_once_annotated_tasks, assign_to_double_annotated_tasks)

        # check that tasks were assigned correctly
        get_task_information_for_annotator(study_project_ids, found_user)


def assign_myself():
    # assign myself to all tasks that are already annotated twice
    print("################################\n")
    email = "TODO"
    all_users = ls.get_users()
    found_users = [u for u in all_users if u.email.lower() == email.lower()]
    me = found_users[0]

    for project_id in study_project_ids:
        project = ls.get_project(project_id)

        filter_double_annotated = Filters.create(Filters.AND, [
            Filters.item(
                Column.total_annotations,
                Operator.EQUAL,
                Type.Number,
                Filters.value(2)
            ),
            Filters.item(
                Column.completed_at,
                Operator.EMPTY,
                Type.Number,
                Filters.value(0)
            ),
            # but don't assign me to reviews that I have already reviewed to prevent any bias
            Filters.item(
                Column.reviewers,
                Operator.NOT_CONTAINS,
                Type.List,
                Filters.value(me.id)
            )
        ])
        found_tasks = project.get_tasks(filter_double_annotated, only_ids=True)
        if not found_tasks:
            print(f'No tasks found for project {project_id}!')
            continue

        project.assign_annotators([me], found_tasks)
        print(f'User {me.username} was assigned to {len(found_tasks)} tasks in project {project_id}')

    get_task_information_for_annotator(study_project_ids, me)


if __name__ == "__main__":
    load_dotenv(dotenv_path="label_studio_api_key.env")

    # Define the URL where Label Studio is accessible and the API key for your user account
    LABEL_STUDIO_URL = 'https://app.heartex.com/'
    API_KEY = os.getenv('LABELSTUDIO_API_KEY')
    # Connect to the Label Studio API and check the connection
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    print(ls.check_connection())

    # IDs in order: Skyrim Paid Mods, AC Unity, Firewatch, Mortal Kombat 11, Borderlands Exclusivity, Ukraine-Russia-RB
    study_project_ids = [53222, 53221, 52584, 52581, 52580, 52575]

    wanted_annotations = 0  # how many tasks a participant wants to annotate; this must always be updated!
    number_tasks_per_project = wanted_annotations // len(study_project_ids)
    # the identifier of the user; this must always be updated!
    user_email = ""

    # TODO die noch nicht annotierten nehmen (noch 20 pro Projekt)
    assign_reviews(is_first_assignment=True, assign_to_once_annotated_tasks=False,
                   assign_to_double_annotated_tasks=False)

    # export_reviews_from_projects(study_project_ids, export_format="CSV")
