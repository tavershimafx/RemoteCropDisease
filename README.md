# RemoteCropDisease
The remote crop disease application is an ML disease detection application for crops. The application has been designed for the dji tello drone which
serves as the data collection device which is then sent to the connected PC for analysis.

The PC application has been trained to identify diseases of over 20 crops.

## Control and operation
In order to begin operation, you need to connect the PC to the drone's wifi network. When connected, you can use the start/connect button present
in the application UI to connect to the drone.

Since the application uses a PS2 game controller, the controller is configured to use the start button to takeoff and the select button to land safely
Or any other manuever key to takeoff. Once the drone has taken off, you can use the right side buttons (shapes buttons) to control the movment of the drone.

### Joystick
The application uses the `pygame` module for joystick control. To initialize the joystick, you call `pygame.init()` thereafter creating an instance of the 
joystick from the available joysticks passing their indexes.

### Joystick events
When a joystick button or axis is operated, the joystick fires some events in respect to the type of event: `JOYAXISMOTION` `JOYBALLMOTION` `JOYBUTTONDOWN` `JOYBUTTONUP` `JOYHATMOTION`
For a list of all events visit [pygame.org](https://www.pygame.org/docs/ref/event.html#pygame.event.Event)

### Event Id for PS2 joystick

| SN | SYMBOL/KEY | EVENT ID |  
|----|------------|----------|
| 1 | Triangle | 0 |
| 2 | Circle | 1 |
| 3 | Times | 2 |
| 4 | Square | 3 |
| 5 | Left 1 | 4 |
| 6 | Right 1 | 5 |
| 7 | Left 2 | 6 |
| 8 | Right 2 | 7 |
| 9 | Select | 8 |
| 10 | Start | 9 |
| 11 | Left steering button | 10 |
| 12 | Right steering button | 11 |
| 13 | Mode | 12 |
