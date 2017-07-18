activity_ordering = [
    8, 20, 21, 22, 23,  # Lying
    7,  # Sitting
    18,  # Transport (sitting)
    6,  # Standing
    15,  # Heel drop
    7,  # Commute (standing)
    3,  # Shuffling
    17,  # Non-vigorous activity
    10,  # Bending
    11,  # Picking
    1,  # Walking
    5,  # Stairs (down)
    4,  # Stairs (up)
    16,  # Vigorous activity
    2,  # Running
    13,  # Cycling (sit)
    14,  # Cycling (stand)
    9,  # Transition
    12,  # Undefined
    0,  # none
]
label_to_number_dict = {
    "none": 0,
    "walking": 1,
    "running": 2,
    "shuffling": 3,
    "stairs (ascending)": 4,
    "stairs (descending)": 5,
    "standing": 6,
    "sitting": 7,
    "lying": 8,
    "transition": 9,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling (sit)": 13,
    "cycling (stand)": 14,
    "heel drop": 15,
    "vigorous activity": 16,
    "non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "lying (prone)": 20,
    "lying (supine)": 21,
    "lying (left)": 22,
    "lying (right)": 23
}
number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])
user_tug_walk_results = {
    "H": {"walk": 4.90, "tug": 8.1, "gender": None, "age": 55, "height": 174, "weight": 82},
    "S01": {"walk": 11.18, "tug": 17.29, "gender": "F", "age": 37, "height": 172, "weight": 67.5},
    "S02": {"walk": 22.87, "tug": 25.12, "gender": "F", "age": 49, "height": 171, "weight": 68},
    "S03": {"walk": 10.07, "tug": 12.56, "gender": "F", "age": 65, "height": 168, "weight": 79},
    "S05": {"walk": 6.71, "tug": 10.66, "gender": "M", "age": 44, "height": 186, "weight": 86},
    "S06": {"walk": 17.56, "tug": 20.64, "gender": "M", "age": 61, "height": 179, "weight": 95},
    "S07": {"walk": 11.19, "tug": 10.39, "gender": "M", "age": 51, "height": 172, "weight": 74},
    "S08": {"walk": 12.32, "tug": 19.14, "gender": "M", "age": 49, "height": 180, "weight": 95},
    "S09": {"walk": 6.97, "tug": 9.20, "gender": "M", "age": 60, "height": 183, "weight": 84},
    "S10": {"walk": 12.38, "tug": 16.45, "gender": "M", "age": 72, "height": 180, "weight": 70},
    "S11": {"walk": 8.31, "tug": 8.68, "gender": "M", "age": 38, "height": 171, "weight": 83},
    "S12": {"walk": 17.68, "tug": 28.92, "gender": "F", "age": 65, "height": 167, "weight": 82},
# Weight not noted. Therefore, average
    "S13": {"walk": 5.80, "tug": 10.89, "gender": "M", "age": 68, "height": 176, "weight": 92},
    "S14": {"walk": 4.68, "tug": 7.80, "gender": "M", "age": 60, "height": 180, "weight": 100},
    "S15": {"walk": 26.84, "tug": 28.20, "gender": "F", "age": 58, "height": 154, "weight": 52},
    "S16": {"walk": 13.62, "tug": 17.94, "gender": "M", "age": 53, "height": 178, "weight": 79}
}
