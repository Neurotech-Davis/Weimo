
def main():
    '''
    continuously run the lidar, gathering data
    continuously get eeg data
    builds walls and map as you go
    should make a separate classifier that does only idle vs jaw clench that always runs
    regardless of eye tracker input so that emergency stop always works but with 
    an easier classification task

    if (eye tracker == point on screen and classifier == go)
      short circut evaluation will make sure classifier only runs 
      if eye tracker already is valid
      use the eye tracker data to pathfind 
      return the output of this pathfinding
    '''
    



    return

if __name__ == "__main__":
    main()