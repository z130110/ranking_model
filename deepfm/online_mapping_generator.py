import json

def load_json2_mapping(loaded_json):
    mapping = {}
    for key_, value_ in loaded_json.items():
        if key_ == "unknown":
            pass
        else:
            mapping[int(key_)] = int(value_)
    return mapping

position_map_load = json.load(open("mapping_dict_sparse/position_map_with_virtual.json"))
position_map = load_json2_mapping(position_map_load)

city_map_load = json.load(open("mapping_dict_sparse/city_map.json"))
city_map = load_json2_mapping(city_map_load)

major_map_load = json.load(open("mapping_dict_sparse/major_map.json"))
major_map = load_json2_mapping(major_map_load)

geek_wordyear_load = json.load(open("mapping_dict_dense/geek_workyear.json"))
geek_wordyear = load_json2_mapping(geek_wordyear_load)

geek_degree_load = json.load(open("mapping_dict_dense/degree.json"))
geek_degree = load_json2_mapping(geek_degree_load)

job_workyear_load = json.load(open("mapping_dict_dense/job_workyear.json"))
job_workyear = load_json2_mapping(job_workyear_load)

job_degree_load = json.load(open("mapping_dict_dense/degree.json"))
job_degree = load_json2_mapping(job_degree_load)

boss_title_type_load = json.load(open("mapping_dict_dense/boss_title_type.json"))
boss_title_type = load_json2_mapping(boss_title_type_load)

converter_id_dict_list = [position_map, city_map, major_map, position_map, city_map]
converter_nonid_dict_list = [geek_wordyear, geek_degree, job_workyear, job_degree, boss_title_type]
all_col_converter = converter_id_dict_list + converter_nonid_dict_list

all_feat_code = [118, 120, 400, 125, 128, 179, 210, 130, 171, 355]

num_lines =  sum([len(col_converter.items()) for col_converter in  all_col_converter])

with open("mapping_dict_online/mapping_online.txt", "w") as f_txt:
    line_counter = 0
    assert len(all_feat_code) == len(all_col_converter)
    for i in range(len(all_feat_code)):
        feat_code = all_feat_code[i]
        col_converter = all_col_converter[i]
        for k, v in col_converter.items():
            line_counter += 1
            if line_counter == num_lines:
                content = str(feat_code) + ":" + str(k) + ":" + str(v)
            else:
                content = str(feat_code) + ":" + str(k) + ":" + str(v) + "\n"
            f_txt.write(content)
