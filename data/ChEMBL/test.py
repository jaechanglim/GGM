if __name__ == "__main__":
    with open("./data_normalized.txt", "w") as newFile:
        values = []
        with open("./data.txt") as oldFile:
            for line in oldFile:
                data = line.split(" ");
                value = float(data[1])
                values.append(value)
        maxValue = max(values)
        minValue = min(values)
        with open("./data.txt") as oldFile:
            for line in oldFile:
                data = line.split(" ")
                value = float(data[1])
                normalizedValue = (value - minValue) / (maxValue - minValue)
                data[1] = str(normalizedValue)
                newLine = " ".join(data)
                newFile.write(newLine)

