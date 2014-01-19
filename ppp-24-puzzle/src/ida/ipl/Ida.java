package ida.ipl;

import ibis.ipl.*;
import java.util.LinkedList;
import java.util.Arrays;
import java.io.IOException;



final class Ida {

  IbisCapabilities ibisCapabilities;
  PortType receivePort;
  PortType clientPort;
  PortType sendPort;

  private class Message{
    IbisIdentifier id;
    int steps;

    Message(IbisIdentifier id, int steps){
      this.id = id;
      this.steps = steps;
    }
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

	private static int solve(Board board, boolean useCache) {
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
		} while (solutions == 0);

		System.out.println("\nresult is " + solutions + " solutions of "
				+ board.bound() + " steps");

    return board.bound();

	}

  private void server(Ibis ibis, String[] args) throws Exception {
    String fileName = null;
		boolean cache = true;
		/* Use suitable default value. */
		int length = 103;
    int counter = 0;


    int numberOfNodes = ibis.registry().getPoolSize();
    LinkedList<Board> workQueue;
    LinkedList<IbisIdentifier> nodes = new LinkedList<IbisIdentifier>();
    LinkedList<Integer> responses = new LinkedList<Integer>();
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
    solutions = new int[workQueue.size()];

    ReceivePort receiver = ibis.createReceivePort(receivePort, "server");
    receiver.enableConnections();
    SendPort sender = ibis.createSendPort(sendPort);

    while(!workQueue.isEmpty()){
      ReadMessage r = receiver.receive();
      Message m = (Message)(r.readObject());
      if(m.steps >= 0){
        solutions[counter] = m.steps;
        counter++;
      }
      sender.connect(m.id, m.id.name());
      WriteMessage w = sender.newMessage();
      w.writeObject(workQueue.pop());
      w.finish();
    }

    //When all boards have been solved, send null
    ReadMessage r = receiver.receive();
    Message m = (Message)(r.readObject());
    if(m.steps >= 0){
      solutions[counter] = m.steps;
      counter++;
    }
    sender.connect(m.id, m.id.name());
    WriteMessage w = sender.newMessage();
    w.writeObject(null);
    w.finish();

    receiver.close();
    sender.close();

    int min = Integer.MAX_VALUE;
    for(int i = 0;i<solutions.length;i++){
      if(solutions[i] < min){
        min = solutions[i];
      }
    }

		System.out.println("\nresult is" + min  + " steps");
    
  }
  
  private void client(Ibis ibis, IbisIdentifier server) throws Exception {
    Board board;
    int steps = 0;

    SendPort sender  = ibis.createSendPort(clientPort);
    sender.connect(server, "server");

    ReceivePort reciever = ibis.createReceivePort(clientPort, "receivePort");
    reciever.enableConnections();

    
    //Send id to server, asking for work
    WriteMessage w = sender.newMessage();
    w.writeObject(new Message(ibis.identifier(), -1));
    w.finish();
    //wait for response
    ReadMessage r = reciever.receive();
    board = (Board)(r.readObject());

    while(board != null){
      steps = solve(board,true);
      w = sender.newMessage();
      w.writeObject(new Message(ibis.identifier(), steps));
      w.finish();
      //wait for response
      r = reciever.receive();
      board = (Board)(r.readObject());
    }

    reciever.close();
    sender.close();
  }

	public void main(String[] args) {
    Ibis ibis = null;
    IbisIdentifier server = null;

    /* Set Ibis Settings*/
    IbisCapabilities ibisCapabilities = new IbisCapabilities(IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.CLOSED_WORLD);
    PortType sendPort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_POLL, PortType.CONNECTION_ONE_TO_MANY);

    PortType receivePort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_POLL, PortType.CONNECTION_ONE_TO_MANY);

    PortType clientPort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_POLL, PortType.CONNECTION_ONE_TO_ONE);




    //Create an Ibis instance
    try{
      ibis = IbisFactory.createIbis(ibisCapabilities, null, sendPort, receivePort, clientPort);
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
}
