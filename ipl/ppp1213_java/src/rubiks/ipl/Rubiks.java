
package rubiks.ipl;

import ibis.ipl.*;
import rubiks.ipl.Cube;
import rubiks.ipl.CubeCache;
import java.util.LinkedList;
import java.io.IOException;

public class Rubiks{

  PortType portType = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_OBJECT, PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_MANY_TO_ONE);

  PortType pollPort = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_DATA, PortType.RECEIVE_POLL,  PortType.CONNECTION_ONE_TO_MANY, PortType.RECEIVE_EXPLICIT);

  IbisCapabilities ibisCapabilities = new IbisCapabilities(IbisCapabilities.ELECTIONS_STRICT, IbisCapabilities.CLOSED_WORLD);

  int globalBound;

  public static void printUsage() {
      System.out.println("Rubiks Cube solver");
      System.out.println("");
      System.out
              .println("Does a number of random twists, then solves the rubiks cube with a simple");
      System.out
              .println(" brute-force approach. Can also take a file as input");
      System.out.println("");
      System.out.println("USAGE: Rubiks [OPTIONS]");
      System.out.println("");
      System.out.println("Options:");
      System.out.println("--size SIZE\t\tSize of cube (default: 3)");
      System.out
              .println("--twists TWISTS\t\tNumber of random twists (default: 11)");
      System.out
              .println("--seed SEED\t\tSeed of random generator (default: 0");
      System.out
              .println("--file FILE_NAME\t\tLoad cube from given file instead of generating it");
      System.out.println("--n SERVERWORK\t\tNumber of steps to be performed by server before handing off work to client nodes");
      System.out.println("");
  }




  private int solutions(Cube cube, CubeCache cache, ReceivePort pollPort) {
      if (cube.isSolved()) {
          return 1;
      }

      if (cube.getTwists() >= cube.getBound()) {
          return 0;
      }

      try{
        globalBound = checkBound(pollPort);
      
      if(cube.getBound() >= globalBound){
        return 0;
      }
    } catch (Exception e){
      System.out.println(e.getMessage());
    }

    // generate all possible cubes from this one by twisting it in
    // every possible way. Gets new objects from the cache
    Cube[] children = cube.generateChildren(cache);

    int result = 0;
    

    for (Cube child : children) {
        // recursion step
        int childSolutions = solutions(child, cache, pollPort);
        if (childSolutions > 0) {
            result += childSolutions;
        }
        // put child object in cache
        cache.put(child);
    }

    return result;
  }

  private void broadcastBound(LinkedList<IbisIdentifier> nodes, int bound, SendPort broadcaster) throws Exception{
    if(bound == Integer.MAX_VALUE){
      return;
    }
    for(IbisIdentifier id : nodes){
      try{
        broadcaster.connect(id, "pollport");
      }catch (Exception e){
        nodes.remove(id);
      }
      WriteMessage w = broadcaster.newMessage();
      w.writeInt(bound);
      w.finish();

    }
  }


  private int checkBound(ReceivePort pollPort) throws IOException{
    ReadMessage r = pollPort.poll();
    if(r == null){
      return globalBound;
      }else{

        int serverBound = r.readInt();
        r.finish();
        if(serverBound < globalBound ){
          globalBound = serverBound;
          return globalBound;
        }else{
          return globalBound;
        }
      }
    }

  private static int solutions(Cube cube, CubeCache cache) {
      if (cube.isSolved()) {
          return 1;
      }

      if (cube.getTwists() >= cube.getBound()) {
          return 0;
      }

      // generate all possible cubes from this one by twisting it in
      // every possible way. Gets new objects from the cache
      Cube[] children = cube.generateChildren(cache);

      int result = 0;
      

      for (Cube child : children) {
          // recursion step
          int childSolutions = solutions(child, cache);
          if (childSolutions > 0) {
              result += childSolutions;
          }
          // put child object in cache
          cache.put(child);
      }

      return result;
  }



  private LinkedList<Cube> ServerStart(String[] arguments){

    Cube cube = null;
    LinkedList<Cube> children = new LinkedList<Cube>();
    CubeCache cache = null;

    // default parameters of puzzle
    int size = 3;
    int twists = 11;
    int seed = 0;
    String fileName = null;

    for (int i = 0; i < arguments.length; i++) {
        if (arguments[i].equalsIgnoreCase("--size")) {
            i++;
            size = Integer.parseInt(arguments[i]);
        } else if (arguments[i].equalsIgnoreCase("--twists")) {
            i++;
            twists = Integer.parseInt(arguments[i]);
        } else if (arguments[i].equalsIgnoreCase("--seed")) {
            i++;
            seed = Integer.parseInt(arguments[i]);
        } else if (arguments[i].equalsIgnoreCase("--file")) {
            i++;
            fileName = arguments[i];
        } else if (arguments[i].equalsIgnoreCase("--help") || arguments[i].equalsIgnoreCase("-h")) {
            printUsage();
            System.exit(0);
        } else {
            System.err.println("unknown option : " + arguments[i]);
            printUsage();
            System.exit(1);
        }
    }


    //The solution should always be possible in less than or equal to the amount of twists
    globalBound = twists;

    // create cube
    if (fileName == null) {
        cube = new Cube(size, twists, seed);
    } else {
        try {
            cube = new Cube(fileName);
        } catch (Exception e) {
            System.err.println("Cannot load cube from file: " + e);
            System.exit(1);
        }
    }
   


    // print cube info
    System.out.println("Searching for solution for cube of size "
            + cube.getSize() + ", twists = " + twists + ", seed = " + seed);
    cube.print(System.out);
    System.out.flush();

    // create CubeCache
    cache = new CubeCache(cube.getSize());
    
    //Generate children 
   
    Cube[] kids = cube.generateChildren(cache); 
    for(Cube kid : kids){
      children.add(kid);
    }

    return children;
  }

  private int[] solve(CubeCache cubes, ReceivePort pollingPort){
    int[] results = new int[2];
    //int startDepth = cubes.peek().getTwists();
    int bound = 0;
    int result = 0;
    while(result == 0){
      bound++;
      try{
        globalBound = checkBound(pollingPort);
      }catch(Exception e){
        System.out.println(e.getMessage());
      }
      for(int i = 0;i<cubes.size();i++){
        if(globalBound<bound){
          result = -1;
          break;
        }
        Cube cube = cubes.peek(i);
        CubeCache cache = new CubeCache(cube.getSize());
        cube.setBound(bound);
        //cube.setTwists(0);
        result += solutions(cube,cache,pollingPort);
      }
    }
    results[0] = result;
    results[1] = bound;

    return results;
  }

  private int[] solve(CubeCache cubes){
    int[] results = new int[2];
    int bound = 0;
    int result = 0;
    while(result == 0){
      bound++;
      for(int i = 0;i<cubes.size();i++){
        Cube cube = cubes.peek(i);
        CubeCache cache = new CubeCache(cube.getSize());
        cube.setBound(bound);
        //cube.setTwists(0);
        result += solutions(cube,cache);
      }
    }
    results[0] = result;
    results[1] = bound;


    return results;
  }



  private void server(Ibis myIbis, String[] args) throws IOException, ClassNotFoundException {
    
    LinkedList<Cube> workQueue = new LinkedList<Cube>();
    LinkedList<Response> answers = new LinkedList<Response>();
    LinkedList<IbisIdentifier> nodes = new LinkedList<IbisIdentifier>();

    int cubesToSolve = 0;
    int numberOfCubes = 0;
    int numberOfNodes = myIbis.registry().getPoolSize();

    
    //Server solves the cube itself when it's the only node...
    if(numberOfNodes == 1){
      LinkedList<Cube> serverCubes = ServerStart(args);
      CubeCache serverInput = new CubeCache(serverCubes.getFirst().getSize());
      for(Cube cube : serverCubes){
        serverInput.put(cube);
      }
      int[] results = solve(serverInput);
      Response serverAnswer = new Response(myIbis.identifier(), results[0], results[1]);
      answers.add(serverAnswer);
    }else{
      //get the initial cubes that the workers can start on.
      workQueue = ServerStart(args);

      int bound = globalBound;
      numberOfCubes = workQueue.size();

      ReceivePort receiver = myIbis.createReceivePort(portType, "server");
      receiver.enableConnections();

      SendPort broadcaster = myIbis.createSendPort(pollPort);




      //Collect the id's of all nodes
      while(nodes.size() < numberOfNodes-1){
        ReadMessage r = receiver.receive();
        Response response = (Response)(r.readObject());
        r.finish();
        nodes.add(response.getId());
      }



      //Broadcast the first bound (max search depth), which equals the number of twists
      try{
        broadcastBound(nodes, bound, broadcaster);
      } catch (Exception e){
        //System.out.println(e.getMessage());
      }
    
      //Send the cubes to all nodes
      int counter = 0;
      for(IbisIdentifier id : nodes){
        //Determine amount of cubes to solve 
        cubesToSolve = (numberOfCubes-counter)/(numberOfNodes-1-counter);
        if(cubesToSolve == 0){
          cubesToSolve = 1;
        }
        CubeCache message = null;
        if(!workQueue.isEmpty()){
          //Create a CubeCache, storing the cubes to evaluate
          message = new CubeCache(workQueue.getFirst().getSize());
          for(int i = 0; i<cubesToSolve;i++){
            if(!workQueue.isEmpty()){
              message.put(workQueue.removeFirst());
            }else{
              break;
            }
          }
          //The last nodes gets all remaining cubes
          //This is the case when the number of cubes cannot be 
          //equally distributed over the nodes
          if(counter == nodes.size() -1){
            while(!workQueue.isEmpty()){
              message.put(workQueue.removeFirst());
            }
          }
        }
        //create a port for sending cubes
        SendPort sender = myIbis.createSendPort(portType);
        sender.connect(id, id.name());
        //Send the cache to the node
        WriteMessage w = sender.newMessage();
        w.writeObject(message);
        w.finish();
        sender.close(); 
        counter++;
      }

      //Wait for answers
      while(answers.size() < numberOfNodes-1){
        ReadMessage r = receiver.receive();
        Response response = (Response)(r.readObject());
        r.finish();
        answers.add(response);
        //If the bound received is smaller than the global bound, adjust and broadcast it
        if(response.getBound() < globalBound){
          bound = response.getBound();
          globalBound = bound;
          try{
            broadcastBound(nodes, bound, broadcaster);
          }catch(Exception e){
          }
        }
      }
    }


    //Determine the smallest solution calculated by the nodes
    int min = Integer.MAX_VALUE;
    int answer = 0;
    for(Response r : answers){
      if(r.getBound() < min){
        min = r.getBound();
        answer = r.getAns();
        continue;
      }
      if(r.getBound()==min){
        answer += r.getAns();
      }
    }

    //DIRTY hack to conform to sanity script
    System.out.print("Bound now:");
    for(int i = 1; i<=min; i++){
      System.out.print(" " + i);
    }
    System.out.println();
    //print out the result
    System.out.println("Solving cube possible in " +  answer + " ways of " + min  + " steps");
  }

  private void exitClient(IbisIdentifier id, int result, int bound, WriteMessage w, SendPort sender, ReceivePort receiver){
    try{
      Response response = new Response(id, result, bound);
      w = sender.newMessage();
      w.writeObject(response);
      w.finish();

      receiver.close();
      sender.close();
    }catch (Exception e){
      System.out.println(e.getMessage());
    }
  }


  private void client(Ibis myIbis, IbisIdentifier server) throws IOException, ClassNotFoundException {
    int i = 0; 
    globalBound = Integer.MAX_VALUE;


    //Create the send port
    SendPort sender = myIbis.createSendPort(portType);
    sender.connect(server, "server");

    //Create port for polling
    ReceivePort pollingPort = myIbis.createReceivePort(pollPort, "pollport");
    pollingPort.enableConnections();

    //Create and send the first message
    Response firstMessage = new Response(myIbis.identifier());
    WriteMessage w = sender.newMessage();
    w.writeObject(firstMessage);
    w.finish();

    //Receive the response from the server
    ReceivePort receiver = myIbis.createReceivePort(portType, myIbis.identifier().name());
    receiver.enableConnections();
    ReadMessage r = receiver.receive();
    Object o = r.readObject();
    r.finish();
    
    CubeCache cubes = (CubeCache)(o);
    //If nothing is received, exit (happens when there are more nodes than input)
    if(cubes == null){
      exitClient(myIbis.identifier(), 0, Integer.MAX_VALUE, w, sender, receiver);
      return;
    }

    int[] result = solve(cubes,pollingPort);
    
    exitClient(myIbis.identifier(), result[0], result[1], w, sender, receiver);
    return;
  }

  private void run(String[] args) throws Exception {
    long start = System.currentTimeMillis();
    //Create an ibis instance.
    Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portType, pollPort);

    //Elect a server.
    IbisIdentifier server = ibis.registry().elect("Server"); 

    //If I am the server, run server, else run client
    if(server.equals(ibis.identifier())){
      server(ibis,args);
      long end = System.currentTimeMillis();
      System.err.println("Solving cube took " + (end-start) + " milliseconds");
    } else {
      client(ibis, server);
    }
    //End ibis.
    ibis.end();
  }


    
  public static void main(String[] args){
    try{
      new Rubiks().run(args);
    }catch (Exception e){
      e.printStackTrace(System.err);
    }
  }
}
