import geo_dataread.gps_read as gpsr


def main():
    print("TEST gps_read")

    data, const = gpsr.read_gps_data("VMEY")
    print(data)
    print(const)


if __name__ == "__main__":
    main()
