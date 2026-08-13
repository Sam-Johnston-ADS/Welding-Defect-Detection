import os

def generate_directory_tree(root_path, output_file="directory_structure.txt", max_files=5):
    """
    Generates the directory tree of a folder and saves it to a text file.
    Limits the number of files printed per directory to avoid huge lists of dataset files.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        def tree(path, prefix=""):
            try:
                all_items = sorted(os.listdir(path))
            except PermissionError:
                return  # Skip directories without read permissions

            # Separate directories and files to apply the limit only to files
            dirs = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
            files = [item for item in all_items if os.path.isfile(os.path.join(path, item))]

            # Truncate the file list if it exceeds the max_files limit
            if len(files) > max_files:
                files = files[:max_files] + ["..."]

            # Combine them: Directories first, then files
            items = dirs + files

            if not items:
                return

            pointers = ["├── "] * (len(items) - 1) + ["└── "]

            for pointer, item in zip(pointers, items):
                f.write(prefix + pointer + item + "\n")

                # If the item is a directory and not our injected "...", recurse into it
                if item != "...":
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        extension = "│   " if pointer == "├── " else "    "
                        tree(full_path, prefix + extension)

        f.write(os.path.abspath(root_path) + "\n")
        tree(root_path)

    print(f"\nDirectory structure saved to '{output_file}'")


if __name__ == "__main__":
    # Enter your folder path here
    folder_path = input("Enter the folder path: ").strip()

    if os.path.exists(folder_path):
        # You can adjust 'max_files' to show more or fewer files per folder
        generate_directory_tree(folder_path, max_files=5)
    else:
        print("Invalid folder path!")