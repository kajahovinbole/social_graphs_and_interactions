import pandas as pd

selected = pd.read_csv("data_lyrics/hentet_recordings_2/top_recordings.csv")
lyrics = pd.read_csv("data_lyrics/hentet_lyrics_2/lyrics.csv")

print("Selected rows:", len(selected))
print("Unique selected MBIDs:", selected["recording_mbid"].nunique())
print("Lyrics rows:", len(lyrics))
print("Unique lyrics MBIDs:", lyrics["recording_mbid"].nunique())

# MBIDs appearing for multiple artists
duplicates = selected[
    selected.duplicated("recording_mbid", keep=False)
].sort_values("recording_mbid")

print(duplicates[["recording_mbid", "artist"]])