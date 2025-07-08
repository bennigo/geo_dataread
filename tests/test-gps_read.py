import geo_dataread.gps_read as gpsr
import gps_plot.timesmatplt as gplot


def main():
    print("TEST gps_read")

    sta = "FAGD"

    comp = ["north", "east", "up"]
    Dcomp = ["Dnorth", "Deast", "Dup"]
    data, const = gpsr.read_gps_data(
        sta,
        detrend_periodic=True,
        detrend_line=False,
        fit=False,
    )
    print(data)
    print(const)
    print("===================================")

    fig = gplot.stdTimesPlot(
        data["yearf"].to_numpy().T,
        data[comp].to_numpy().T,
        data[Dcomp].to_numpy().T,
        Title=sta,  # start=toDateTime([2015.6])[0]
    )

    gpsr.save_detrend_const(const, detrendFile="tmp.csv")

    filename = f"{sta}-check"
    gplot.saveFig(filename, "pdf", fig)


if __name__ == "__main__":
    main()
