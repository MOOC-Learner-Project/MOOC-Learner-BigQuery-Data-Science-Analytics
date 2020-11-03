#! /usr/bin/env python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=""

from google.cloud import bigquery
from google.cloud import storage

import json
import gzip
import uuid
import time


def export_data_to_gcs(project, dataset_name, dest):
    """ Function to export BigQuery tables as JSON files
    :param project: the BigQuery project
    :param dataset_name: the particular dataset within the project
    :param dest: the export destination
    """
    bigquery_client = bigquery.Client(project)
    dataset = bigquery_client.dataset(dataset_name)
    tables = dataset.list_tables()
    for i, table in enumerate(tables):
        table.reload()
        print('Table number:' + str(i))
        print('Numer of rows:' + str(table.num_rows))
        tb = dataset.table(table.name)
        destination = dest + str(i) + '_*.json.gz'
        job_name = str(uuid.uuid4())
        job = bigquery_client.extract_table_to_storage(job_name, tb, destination)
        job.destination_format = 'NEWLINE_DELIMITED_JSON'
        job.compression = 'GZIP'
        job.begin()
        wait_for_job(job)
        print('Exported {}:{} to {}'.format(dataset_name, table.name, destination))


def download_files_from_gcs(bucket_name, dir_path, terms, files, save_filename):
    """ Function to export specific JSON files to alfad sever
    :param project: the GCS bucket
    :param terms: list of course term strings to download
    :param files: list of JSON.GZ filename strings to download
    :param save_filename: string of filename to save to
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    for t in terms:
        path = dir_path.format(t)
        for f in files:
            source_blob_name = '{}_latest/{}'.format(t, f)
            blob = bucket.blob(source_blob_name)
            dest_file_name = '{}/{}'.format(path, f)
            blob.download_to_filename(dest_file_name)
            print('Blob {} downloaded to {}.'.format(source_blob_name, dest_file_name))

        combine_json(files, path, save_filename)

def combine_json(read_files, dir_path, outf):
    """
    :param read_files: list of json.gz files to combine
    :param outf: string name of output combined json file
    """
    with open('{}/{}'.format(dir_path, outf), 'w') as outfile:
        outfile.write('{}'.format(
            ''.join([gzip.open('{}/{}'.format(dir_path, f), 'rb').read().decode('utf-8') for f in read_files])))


def wait_for_job(job):
    """ Function to wait on a job
    :param job: the specified job
    """
    while True:
        job.reload()
        print (job.state)
        if job.state == 'DONE':
            if job.error_result:
                raise RuntimeError(job.errors)
                return
        print('sleep')
        time.sleep(1)


if __name__ == '__main__':
    
    download_files_from_gcs(bucket_name='bucket_name',
        dir_path='save_path',
        terms=['Course_name'],
        files=['file1.json.gz'],
        save_filename='file1.json')


