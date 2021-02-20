import csv


def data_remover():
    # transform txt input into csv to be able to work with it more comfortably
    with open('apartmentComplexData.txt', 'r') as input_txt:
        stripped = (line.strip() for line in input_txt)
        lines = (line.split(",") for line in stripped if line)
        with open('log.csv', 'w') as input_csv:
            writer = csv.writer(input_csv)
            writer.writerows(lines)

    # open the converted csv file
    with open('log.csv', 'r') as source:
        rdr = csv.reader(source)
        # write only indices 2,3,4,5,6,8 that correspond to columns 3,4,5,6,7,9
        with open('dataset.csv', 'w') as result:
            wtr = csv.writer(result)
            for r in rdr:
                wtr.writerow((r[2], r[3], r[4], r[5], r[6], r[8]))

    # convert back to txt (if needed)
    # with open('data.txt', 'w') as my_output_file:
    #     with open('dataset.csv', 'r') as my_input_file:
    #         [my_output_file.write(" ".join(row) + '\n') for row in csv.reader(my_input_file)]
    #     my_output_file.close()
