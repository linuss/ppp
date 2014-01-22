package ida.ipl;

import ibis.ipl.*;
import java.util.LinkedList;
import java.util.Arrays;
import java.io.IOException;
import java.io.Serializable;



final class Ida implements MessageUpcall {
  static PortType receivePort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.CONNECTION_MANY_TO_ONE, PortType.RECEIVE_EXPLICIT);
  static PortType sendPort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_ONE_TO_ONE);
  static PortType clientSendPort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_MANY_TO_ONE);
  static PortType clientReceivePort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_ONE_TO_ONE);
  static PortType broadcastport = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_DATA, PortType.RECEIVE_AUTO_UPCALLS, PortType.CONNECTION_ONE_TO_MANY);
  static IbisCapabilities ibisCapabilities = new IbisCapabilities(IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.CLOSED_WORLD);
  int globalBound = Integer.MAX_VALUE;

  private static class Message implements Serializable{
    IbisIdentifier id;
    int steps;
    int solutions;

    Message(IbisIdentifier id, int steps, int solutions){
      this.id = id;
      this.steps = steps;
      this.solutions = solutions;
    }

    Message(IbisIdentifier id, int steps){
      this.id = id;
      this.steps = steps;
      this.solutions = 0;
    }
  }

  public void upcall(ReadMessage message) throws IOException{
    globalBound = message.readInt();
  }
    



	/**
	 * expands this board into all possible positions, and returns the number of
	 * solutions. Will cut off at the bound set in the board.
	 */
	private static int solutions(Board board, BoardCache cache) {
		if (board.distance() == 0) {
			return 1;
		}

		if (board.distance() > board.bound()) {
			return 0;
		}

		Board[] children = board.makeMoves(cache);
		int result = 0;

		for (int i = 0; i < children.length; i++) {
			if (children[i] != null) {
				result += solutions(children[i], cache);
			}
		}
		cache.put(children);
		return result;
	}

	/**
	 * expands this board into all possible positions, and returns the number of
	 * solutions. Will cut off at the bound set in the board.
	 */
	private static int solutions(Board board) {
		if (board.distance() == 0) {
			return 1;
		}

		if (board.distance() > board.bound()) {
			return 0;
		}

		Board[] children = board.makeMoves();
		int result = 0;

		for (int i = 0; i < children.length; i++) {
			if (children[i] != null) {
				result += solutions(children[i]);
			}
		}
		return result;
	}

	private Message solve(Board board, boolean useCache, Ibis ibis) {
		BoardCache cache = null;
		if (useCache) {
			cache = new BoardCache();
		}
		int bound = board.distance();
		int solutions;

		System.out.print("Try bound ");
		System.out.flush();

		do {
			board.setBound(bound);

			System.out.print(bound + " ");
			System.out.flush();

			if (useCache) {
				solutions = solutions(board, cache);
			} else {
				solutions = solutions(board);
			}

			bound += 2;
		} while (solutions == 0 && bound <= globalBound);

		System.out.println("\nresult is " + solutions + " solutions of "
				+ board.depth() + " steps");

    Message result = new Message(ibis.identifier(), board.depth(), solutions);

    return result;

	}

  private static void broadcastBound(SendPort broadcaster, int bound) throws Exception{
    WriteMessage m = broadcaster.newMessage();
    m.writeInt(bound);
    m.finish();
  }

      



  private static void server(Ibis ibis, String[] args) throws Exception {
    String fileName = null;
		boolean cache = true;
		/* Use suitable default value. */
		int length = 103;
    int counter = 0;
    int bound = Integer.MAX_VALUE;


    int numberOfNodes = ibis.registry().getPoolSize();
    LinkedList<Board> workQueue;
    LinkedList<IbisIdentifier> nodes = new LinkedList<IbisIdentifier>();
    int[] solutions;

		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("--file")) {
				fileName = args[++i];
			} else if (args[i].equals("--nocache")) {
				cache = false;
			} else if (args[i].equals("--length")) {
				i++;
				length = Integer.parseInt(args[i]);
			} else {
				System.err.println("No such option: " + args[i]);
				System.exit(1);
			}
		}
		Board initialBoard = null;

		if (fileName == null) {
			initialBoard = new Board(length);
		} else {
			try {
				initialBoard = new Board(fileName);
			} catch (Exception e) {
				System.err
						.println("could not initialize board from file: " + e);
				System.exit(1);
			}
		}
		System.out.println("Running IDA*, initial board:");
		System.out.println(initialBoard);


    //Create work queue
    workQueue = new LinkedList<Board>(Arrays.asList(initialBoard.makeMoves()));
    System.out.printf("There are %d boards in the workqueue\n", workQueue.size());
    solutions = new int[workQueue.size()];

    ReceivePort receiver = null;
    try{
      receiver = ibis.createReceivePort(receivePort, "server");
    }catch (Exception e){
      System.out.println(e.toString());
      System.exit(1);
    }
    receiver.enableConnections();
    System.out.printf("Server listening\n");
    SendPort sender = null;

    SendPort broadcaster = ibis.createSendPort(broadcastport);

    while(!workQueue.isEmpty()){
      ReadMessage r = receiver.receive();
      Message m = (Message)(r.readObject());
      r.finish();
      if(!nodes.contains(m.id)){
        nodes.add(m.id);
        broadcaster.connect(m.id, "broadcastport");
      }
      System.out.println("Server received message from " + m.id + m.id.name());
      if(m.steps >= 0){
        solutions[counter] = m.steps;
        counter++;
        if(m.steps < bound){
          bound = m.steps;
          broadcastBound(broadcaster, bound);
        }
      }
      System.out.println("Server tries to connect to sender");
      sender = ibis.createSendPort(sendPort);
      sender.connect(m.id, "receivePort");
      System.out.printf("Server connected to sender!\n");
      WriteMessage w = sender.newMessage();
      w.writeObject(workQueue.pop());
      w.finish();
      sender.close();
      System.out.printf("Server sent object to sender\n");
    }

    //When all boards have been solved, send null
    for(int i=0;i<numberOfNodes-1;i++){
      ReadMessage r = receiver.receive();
      Message m = (Message)(r.readObject());
      if(m.steps >= 0){
        solutions[counter] = m.steps;
        counter++;
      }
      sender = ibis.createSendPort(sendPort);
      sender.connect(m.id, m.id.name());
      WriteMessage w = sender.newMessage();
      w.writeObject(null);
      w.finish();
    }

    receiver.close();

    int min = Integer.MAX_VALUE;
    for(int i = 0;i<solutions.length;i++){
      if(solutions[i] < min){
        min = solutions[i];
      }
    }

		System.out.println("\nresult is" + min  + " steps");
    ibis.end();
    
  }
  
  private void client(Ibis ibis, IbisIdentifier server) throws Exception {
    Board board;
    Message response;

    SendPort sender  = ibis.createSendPort(clientSendPort);
    sender.connect(server, "server");
    System.out.println("Client created sendport, and connected to server");

    ReceivePort reciever = ibis.createReceivePort(clientReceivePort, "receivePort");
    reciever.enableConnections();
    System.out.println("Client created receiveport");

    ReceivePort receiveBroadcast = ibis.createReceivePort(broadcastport, "broadcastport", this);
    receiveBroadcast.enableConnections();

    
    //Send id to server, asking for work
    WriteMessage w = sender.newMessage();
    w.writeObject(new Message(ibis.identifier(), -1));
    w.finish();
    System.out.println("client sent message to server");
    //wait for response
    System.out.println("client is waiting for response from server");
    ReadMessage r = reciever.receive();
    System.out.println("client has received response!");
    board = (Board)(r.readObject());
    r.finish();

    while(board != null){
      response = solve(board,true, ibis);
      w = sender.newMessage();
      w.writeObject(response);
      w.finish();
      //wait for response
      r = reciever.receive();
      board = (Board)(r.readObject());
      r.finish();
    }

    reciever.close();
    sender.close();
    ibis.end();
  }

  public void start(String[] args){
    Ibis ibis = null;
    IbisIdentifier server = null;



    //Create an Ibis instance
    try{
      ibis = IbisFactory.createIbis(ibisCapabilities, null, sendPort, receivePort, clientReceivePort, clientSendPort, broadcastport);
    }catch (IbisCreationFailedException e){
      System.out.println("Failed to create Ibis!\n" + e.getMessage());
      System.exit(1);
    }

    //Elect server
    try{
      server = ibis.registry().elect("Server");
    }catch (IOException e){
      System.out.println("Failed to elect server!\n" + e.getMessage());
      System.exit(1);
    }
		long start = System.currentTimeMillis();

    if(server.equals(ibis.identifier())){
      try{
        server(ibis, args);
      }catch (Exception e){
        System.out.println("Caught exception in server:" + e.getMessage());
        System.exit(1);
      }
      // NOTE: this is printed to standard error! The rest of the output
      // is
      // constant for each set of parameters. Printing this to standard
      // error
      // makes the output of standard out comparable with "diff"
		  long end = System.currentTimeMillis();
      System.err.println("ida took " + (end - start) + " milliseconds");
    }else{
      try{
        client(ibis,server);
      }catch (Exception e){
        System.out.println("Caught exception in client:" + e.getMessage());
        System.exit(1);
      }
    }
  }

	public static void main(String[] args) {
    new Ida().start(args);
	}
}
