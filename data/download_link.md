The source images for training are stored here.

## Download data

Please download the data from
https://drive.google.com/uc?id=1qELGC5zHnrksd98nqXHLybX3BA8tz2qL

Unzip the data and you will see the folder: `data`. Use it to replace the `data`.

Inside the folder, there is a `valid_images.txt`, which describes the label of each image that I used for training. (For your conviniene, I've included it in this repo, and you can view it at [data/valid_images.txt](valid_images.txt).)

## Data Folder structure

  ```
  data
  ├── running_03-13-11-27-50-720
  ├── running_03-12-09-18-26-176
  ├── running_03-02-12-34-01-795
  ├── jogging_03-02-12-34-01-795
  ├── jogging_03-02-12-36-05-185
  ├── jogging_03-12-09-23-41-176
  ├── walking_03-12-09-13-10-500
  ├── walking_03-02-12-30-23-393
  ├── walking_03-12-09-13-10-875
  ├── clapping_03-08-20-26-57-195
  ├── clapping_03-13-13-21-48-761
  ├── clapping_03-08-20-24-55-587
  ```

## Images for training:

  Number of actions = 4  
  Total training images = 12172
  Number of images of each action:  

  |Label|Number of frames|
  |:---:|:---:|
  running|  2976|  
  jogging| 2874|  
  walking| 3353|  
  clapping|  2969|  