# chat

 a basic client/server anonymous command-line chat program.
 The clients will send messages to the server with their desired username attached.
 The server will broadcast all the messages it receives to all the clients.
 
 # Server Program
 •The server will keep a list of client IP’s and ports.
 •When a message is received it is sent too all existing clients.
 •A log of time and message is printed to screen.
 
 # Client Program
 •The client will ask the user for a username.
 •A separate thread will handle incoming messages.
