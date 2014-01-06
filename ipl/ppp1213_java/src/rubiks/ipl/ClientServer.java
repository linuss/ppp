package rubiks.ipl;

import ibis.ipl.*;
import rubiks.sequential.Cube;
import rubiks.sequential.CubeCache;
import java.util.LinkedList<E>;

public class ClientServer{

  PortType portType = new PortType(PortType.COMMUNICATION_RELIABLE, PortType.SERIALIZATION_DATA, PortType.RECEIVE_EXPLICIT, PortType.CONNECTION_ONE_TO_MANY);

  IbisCapabilities ibisCapabilities = new IbisCapabilities(IbisCapabilties.ELECTIONS_STRICT);


  private static Cube[] ServerStart(String[] arguments){

    Cube cube = null;
    Cube[] children = null;
    CubeCache cache = null;

    // default parameters of puzzle
    int size = 3;
    int twists = 11;
    int seed = 0;
    String fileName = null;

    // number of threads used to solve puzzle
    // (not used in sequential version)

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
   
    // create CubeCache
    cache = new CubeCache(cube.getSize());


    // print cube info
    System.out.println("Searching for solution for cube of size "
            + cube.getSize() + ", twists = " + twists + ", seed = " + seed);
    cube.print(System.out);
    System.out.flush();

    
    //Generate children 
    children = cube.generateChildren(cache);

    return children;
  }



  private void server(Ibis MyIbis, String[] args) throws IO Exception {
    
    Cube[] children = null;
    LinkedList<Cube> workQueue = new LinkedList<Cube>();
    LinkedList<IbisIdentifier> workerQueue = new LinkedList<IbisIdentifier>();
    LinkedList<Response> answers = new LinkedList<Response>();

    //get the initial cubes that the workers can start on.
    children = ServerStart(args);

    for(Cube child : children) {
      workQueue.add(child);
    }
    
    RecievePort receiver = myIbis.createReceivePort(portType, "server");
    receiver.enableConnections();

    while(workQueue.size > 0 && workerQueue.size < myIbis.registry.getPoolSize-1){
      ReadMessage r = reciever.receive();
      IbisIdentifier id = r.readObject();
      r.finish();
      workerQueue.add(id);
      if(workQueue.size > 0){
        //Create a port for responding to the request
        SendPort sender = myIbis.createSendPort(portType);
        sender.connect(workerQueue.getFirst());
        //Send the message (the cube)
        WriteMessage w = sender.newMessage();
        w.writeObject(workQueue.getFirst());
        w.finish();
        //close port
        sender.close();
      }
    }
          

     




  private void run(String args[]) throws Exception {
    //Create an ibis instance.
    Ibis ibis = IbisFactory.createIbis(ibisCapabilities, null, portType);

    //Elect a server.
    IbisIdentifier server = ibis.registry().elect("Server"); 

    //If I am the server, run server, else run client
    if(server.equals(ibis.identifier())){
      server(ibis,args);
    } else {
      client(ibis);
    }
    
    //End ibis.
    ibis.end();
  }

  private

    
  public static void main(String[] args){
    try{
      new ClientServer.run(args);
    }catch (Exception e){
      e.printStackTrace(System.err);
    }
  }
}
