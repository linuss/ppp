package rubiks.ipl;

import ibis.ipl.*;

public class Response implements java.io.Serializable{
  private IbisIdentifier id;
  private int answer;
  private int bound; 


  public Response(IbisIdentifier id){
    this.id = id;
    this.answer = -1;
    this.bound = -1;
  }

  public Response(IbisIdentifier id, int ans, int bnd){
    this.id = id;
    this.answer = ans;
    this.bound = bnd;
  }

  public IbisIdentifier getId(){
    return id;
  }

  public int getAns(){
    return answer;
  }

  public int getBound(){
    return bound;
  }
}

