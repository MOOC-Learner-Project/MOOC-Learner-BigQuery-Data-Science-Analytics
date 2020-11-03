import os
import json
import csv
from preprocess_util import *

class JSON2CSV:
    """Convert BigQuery JSON to CSV tables.
    """

    def __init__(self, read_dir, save_dir, dict_of_fields):
        """Initialize the converter.
        :param read_dir: the directory of JSON files
        :param save_dir: the directory to save CSV files
        :param dict_of_fields: a dictionary of fields of interest in each table
        """
        # Validate
        if not os.path.isdir(read_dir):
            raise NotADirectoryError("{}: read dir does not exist.".format(self.__class__.__name__))
        self.read_dir = read_dir
        if not os.path.isdir(save_dir):
            raise NotADirectoryError("{}: save dir does not exist.".format(self.__class__.__name__))
        self.save_dir = save_dir
        if not isinstance(dict_of_fields, dict) and \
                all([isinstance(k, str) for k in dict_of_fields.keys()]) and \
                all([isinstance(v, list) for v in dict_of_fields.values()]):
            raise ValueError("{}:".format(self.__class__.__name__))

        # Adapter
        self.file_input, self.file_output, self.output_fields, \
        self.pre_map_dict, self.post_map_dict = self.adapter(dict_of_fields)

        # Initialize
        self.typedict = {}

    def adapter(self, dict_of_fields):
        """Generate the required parameters by the dictionary of interested fields.
        Sensitive to username fields, which are configured to be hashed here.
        This is to ensure that the processed CSVs do not contain any sensitive data.
        :param dict_of_fields: a dictionary of fields of interest in each table
        :return: file_input, file_output, output_fields, pre_map_dict, post_map_dict
        """

        # Hash function of username
        def hash_processor(d):
            if 'username' in d:
                return hash_username(d['username'])
            self.report(d, 'username')

        # Natural naming
        file_input = {k: k + '.json' for k in dict_of_fields}
        file_output = {k: k + '.csv' for k in dict_of_fields}
        # Output fields
        output_fields = dict_of_fields
        # Map Dictionary
        pre_map_dict = {k: [
            (
                (0, 0),
                k,
                {f: f for f in v if f != 'username'}
            ),
            (
                (0, 1),
                k,
                {
                    'username': hash_processor
                }
            ) if any([f == 'username' for f in v]) else None
        ] for k, v in dict_of_fields.items()}
        pre_map_dict = {k: [p for p in l if p is not None] for k, l in pre_map_dict.items()}
        post_map_dict = {k: [
            (
                (2, 2),
                k,
                lambda x: x,
            )
        ] for k, v in dict_of_fields.items()}
        return file_input, file_output, output_fields, pre_map_dict, post_map_dict

    def report(self, anything, flag=None):
        """Report missing or malformed entries.
        :param anything: missing or malformed entry.
        :param flag: category it belongs to.
        """
        self.typedict[flag] = self.typedict.get(flag, 0) + 1

    def process(self):
        """Process the JSON files and convert to CSV tables.
        """
        self.saving(self.save_dir,
                    self.file_output,
                    self.concatenating(
                        self.mapping(
                            self.concatenating(
                                self.mapping(
                                    self.loading(
                                        self.read_dir,
                                        self.file_input
                                    ),
                                    self.pre_map_dict
                                )
                            ),
                            self.post_map_dict
                        )
                    ),
                    self.output_fields
                    )
        print("{}: Process finished.".format(self.__class__.__name__))
        print(" - tables converted: {} - missing field types: {}".format(list(self.file_input.keys()),
                                                                         self.typedict))

    def get_fields(self, dicts):
        if not dicts:
            print("not dicts")
            return []
        if not isinstance(dicts, list) \
                or not isinstance(dicts[0], dict):
            raise ValueError("Invalid dicts")
        fields = sorted(list(dicts[0].keys()))
        for d in dicts:
            if sorted(list(d.keys())) != fields:
                raise ValueError("Invalid dicts")
        return fields

    def loading(self, dir_path, file_input):
        if not os.path.isdir(dir_path):
            raise ValueError("Invalid input path")
        dir_path = os.path.abspath(dir_path)
        return {_input: self.load_file(os.path.join(dir_path + '/' + file_input[_input]))
                for _input in file_input}

    def load_file(self, path):
        with open(path, 'r', encoding='utf-8') as json_file:
            # print("loading: ",'/'.split(path)[-1])
            json_lines = json_file.readlines()
            # dicts = [json.loads(x,encoding='utf-8') for x in json_lines]
            dicts = []
            for i, x in enumerate(json_lines):
                try:
                    line_dict = json.loads(x, encoding='utf-8')
                except:
                    raise ValueError('misformatted line {}'.format(i))
                dicts.append(line_dict)
        return dicts

    def saving(self, dir_path, file_output, output_collection, output_fields):
        if not os.path.isdir(dir_path):
            raise ValueError("Invalid input path")
        dir_path = os.path.abspath(dir_path)
        if not all(_output in output_collection for _output in file_output):
            raise ValueError("Insufficient output_dict")
        for _output in file_output:
            self.save_file(os.path.join(dir_path + '/' + file_output[_output]),
                           output_collection[_output],
                           output_fields[_output])

    def save_file(self, path, dicts, fields):
        self.save_csv(path, dicts, fields)

    def save_csv(self, path, dicts, fields):
        if set(fields) != set(self.get_fields(dicts)):
            raise ValueError("Invalid output_fields")
        with open(path, 'w', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                delimiter='\t' if path.endswith('.sql')
                else ',',
                fieldnames=fields,
                quotechar='"',
                escapechar='\\',
                lineterminator='\n')
            writer.writeheader()
            for d in dicts:
                writer.writerow({f: d[f] for f in fields})

    def mapping(self, input_collection, map_dict):
        return {_output: [self.map_unit(
            map_obj[0],
            input_collection[map_obj[1]],
            map_obj[2]
        )
            for map_obj in map_dict[_output]]
            for _output in map_dict}

    def map_unit(self, order, dicts, unit_map):
        if order == (0, 0):
            return self.map_unit_0_0(dicts, unit_map)
        elif order == (0, 1):
            return self.map_unit_0_1(dicts, unit_map)
        elif order == (0, 2):
            return self.map_unit_0_2(dicts, unit_map)
        elif order == (2, 2):
            return self.map_unit_2_2(dicts, unit_map)
        else:
            raise ValueError("Invalid map order")

    def map_unit_0_0(self, dicts, unit_map):
        return [{f: d[unit_map[f]] if unit_map[f] in d else self.report(d, unit_map[f])
                 for f in unit_map}
                for d in dicts]

    def map_unit_0_1(self, dicts, unit_map):
        return [{f: unit_map[f](d)
                 for f in unit_map}
                for d in dicts]

    def map_unit_0_2(self, dicts, unit_map):
        return [{f: unit_map[f](d, dicts)
                 for f in unit_map}
                for d in dicts]

    def map_unit_2_2(self, dicts, unit_map):
        return unit_map(dicts)

    def concatenating(self, mapped_collection):
        return {_output: self.concat_dicts(mapped_collection[_output])
                for _output in mapped_collection}

    def all_disjoint(self, sets):
        union = set()
        for s in sets:
            for x in s:
                if x in union:
                    return False
                union.add(x)
        return True

    def concat_dict(self, list_of_dict):
        if not all(isinstance(d, dict) for d in list_of_dict):
            raise ValueError("Invalid list_of_dict")
        if not self.all_disjoint([set(d.keys()) for d in list_of_dict]):
            raise ValueError("Illegal concatenation of list_of_dict")
        sd = {}
        for d in list_of_dict:
            sd.update(d)
        return sd

    def concat_dicts(self, list_of_dicts):
        if not list_of_dicts:
            return []
        if not all(len(dicts) == len(list_of_dicts[0])
                   for dicts in list_of_dicts):
            raise ValueError("Illegal concatenation of list_of_dicts")
        return [self.concat_dict([list_of_dicts[x][y]
                                  for x in range(len(list_of_dicts))])
                for y in range(len(list_of_dicts[0]))]

