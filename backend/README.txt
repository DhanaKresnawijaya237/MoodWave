PMEmo Dataset - 2019/06



- The dataset consists of:
.
├── metadata.csv, song id, file name, song tile, artists, album, duration, start/end timestamp of chorus
│ 
├── chorus, 794 music clips (.MP3, chorus part)
│   ├── 1.mp3
│   ├── ...
│   └── 13.mp3
│ 
├── annotations, the static and dynamic annotations for 794 songs in the dimension of valence and arousal
│   ├── dynamic_annotations.csv
│   ├── dynamic_annotations_std.csv
│   ├── static_annotations.csv
│   └── static_annotations_std.csv
│ 
├── EDA, the electrodermal activity data of each subject for 794 songs
│   ├── 1000_EDA.csv
│   ├── ...
│   └── 9_EDA.csv
│ 
├── comments, Chinese/English user comments from NetEaseMusic/SoundCloud
│   ├── netease
│   │   ├── 1.txt
│   │   ├── ...
│   │   └── 996.txt
│   └── soundcloud
│       ├── 1.txt
│       ├── ...
│       └── 996.txt
│ 
├── features, pre-computed acoustic features
│   ├── dynamic_features.csv, 260 acoustic low-level descriptors of each 0.5 second for each song
│   └── static_features.csv, 6373 overall acoustic features of each song
│ 
├── lyrics, lyrics files of songs
│   ├── 1.lrc
│   ├── ...
│   └── 996.lrc
│ 
└── netease_soundcloud.csv, music meta of NetEaseMusic/SoundCloud


