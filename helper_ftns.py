from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def get_image_sizes(folder, group):
    records = []

    for path in folder.glob("*"):
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            with Image.open(path) as img:
                width, height = img.size

            records.append({
                "filename": path.name,
                "group": group,
                "width": width,
                "height": height,
                "area_pixels": width * height
            })

    return records


def compare_folder_images(folder1, folder2):
    data = (
            get_image_sizes(folder1, "Folder 1") +
            get_image_sizes(folder2, "Folder 2")
    )

    df = pd.DataFrame(data)

    print(df.groupby("group")[["width", "height", "area_pixels"]].describe())

    for group in df["group"].unique():
        subset = df[df["group"] == group]

        plt.hist(
            subset["area_pixels"],
            bins=30,
            alpha=0.5,
            label=group
        )

    plt.xlabel("Image size (width × height pixels)")
    plt.ylabel("Number of images")
    plt.legend()
    plt.show()