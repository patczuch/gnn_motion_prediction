import os


def delete_files_with_extension(root_folder, extension):
    if not extension.startswith("."):
        extension = "." + extension

    deleted = 0

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(extension):
                full_path = os.path.join(dirpath, filename)
                try:
                    os.remove(full_path)
                    deleted += 1
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")

    print(f"\nDone. Deleted {deleted} file(s).")


if __name__ == "__main__":
    delete_files_with_extension("./datasets", ".bvh.feat.pt")
