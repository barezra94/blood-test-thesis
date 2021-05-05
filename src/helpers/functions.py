def add_invalid_column(data, features_list):
    # If at least one test is not in the valid value range then value of column will be 0 (False)
    data["allTestValid"] = 1

    for key in features_list:
        # Gets data from dictonary
        min_value = features_list[key][0]
        max_value = features_list[key][1]

        # Change values of "allTestValid" according to test values in range
        data.loc[(data[key] > max_value) | (data[key] < min_value), "allTestValid"] = 0
        df = data.loc[(data[key] > max_value) | (data[key] < min_value)]
        print(key, df.shape)

    return data

