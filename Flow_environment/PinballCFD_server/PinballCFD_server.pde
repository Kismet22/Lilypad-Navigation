import java.util.*;
import org.apache.xmlrpc.*;
import java.util.concurrent.*;
import java.util.concurrent.Semaphore;
import com.alibaba.fastjson.JSONObject;
Pinball test;  
SaveScalar dat;
SaveData veldata1;
SaveData veldata2;
SaveData veldata3;
SaveVectorField data;
float maxT;
PrintWriter output;

int Re = 500;
float chord = 1.0;
int resolution = 16, xLengths = 20, yLengths = 8, zoom = 100/resolution;
int plotTime = 1;
int picNum = 20;
int simNum = 1;
float tStep = .0075;
String datapath = "saved" + "/";
int EpisodeNum = 1;
int initTime = 1, callLearn = 16, act_num = 0;
int EpisodeTime = 60, ActionTime = 0; //TODO: actionTime = 5
float CT = 0, CL = 0, CP = 0, forceX = 0;
float AvgCT = 0, Avgeta = 0;
float AoA = 0, A = 0;
float y = 0, angle = 0;
PVector pos = new PVector();
PVector vel = new PVector();
float cost = 0, ref = 5, ave_reward=0; //
String p;
XmlRpcClient client;
WebServer server;
ArrayList<PVector> velvalue = new ArrayList<PVector>();
ArrayList<Float> CV15 = new ArrayList<Float>();
ArrayList<Float> CV30 = new ArrayList<Float>();
ArrayList<Float> CV50 = new ArrayList<Float>();
ArrayList<Float> surfacePressures;
int velcount = 0;
float[] ycoords;

int StepTime = 10, StepCount = 1;// used for adjust Ct avg length
// flow_info
boolean flowFlag = false; // 默认不包含流场数据

float nextactionC=0, nextactionB=0,nextactionA=0,nextactionAvel = 0, nextactionAoAvel = 0;
float formerA = 0, formerAoA = 0,formerAvel = 0, formerAoAvel = 0;
float[] NextAction = {nextactionA,nextactionB,nextactionC};
float[] FormalAction = {formerA,formerAvel,formerAoA, formerAoAvel};

//remote info exchange
ArrayList state = new ArrayList();
float[] action = new float[3];
float reward_buffer = 0;
Boolean done = false;
int reset_flag = 0;
private float[][] flowVelocityX;
private float[][] flowVelocityY;
PVector target = new PVector(150,100);

// Semaphore to provide asynchronization
Semaphore action_sem = new Semaphore(0);
Semaphore state_sem = new Semaphore(0);




void settings()
{
  size(640, 256);

}

// 旧的启动方式，不包含流场数据符号
/*void setup()
{

  try
  {
    server = StartServer(int(args[0])); //端口号
  }
  
  catch (Exception ex)
  {
    println(ex);
  }
  setUpNewSim(EpisodeNum);
}*/


void setup() {
    try {
        if (args != null && args.length > 1) {
            int port = int(args[0]); // 解析端口号
            boolean includeFlow = args[1].equalsIgnoreCase("true"); // 解析流场标志
            server = StartServer(port, includeFlow); // 调用带参数的 StartServer
            println("Server started on port: " + port + " with flow flag: " + includeFlow);
        } else {
            println("Error: Insufficient arguments provided. Please specify a port and flow flag.");
        }
    } catch (Exception ex) {
        println("Error starting server: " + ex);
    }
    setUpNewSim(EpisodeNum); // 初始化仿真
}

void draw() {
  // 始终执行仿真环境的更新
  // 获取远程传来的动作
  NextAction = callActionByRemote();
  nextactionA = NextAction[0];
  nextactionB = NextAction[1];
  nextactionC = NextAction[2];
  act_num += 1;

  // 更新环境状态
  test.xi0 = nextactionA;
  test.xi1 = nextactionB;
  test.xi2 = nextactionC;

  // 更新仿真环境状态并获取流场速度
  // 322 * 130
  test.update2();
  // 记录单步流场速度
  //println("Type of test.flow.u.x.a: " + test.flow.u.x.a.getClass());
  flowVelocityX = test.flow.u.x.a;
  flowVelocityY = test.flow.u.y.a;

  // 获取当前位置、速度和表面压力
  pos = test.pos;
  vel = test.vel;
  surfacePressures = test.surfacePressures;
  String state_cn = multy_state(pos, vel, surfacePressures);
  //println("Lilypad Print:", pos);

  // 更新状态
  state.clear();
  state.add(state_cn);
  // 传输流场速度信息
  // flowVelocityjson = flowFieldToJson(flowVelocity);

  // 释放状态信号量，通知其他线程状态已更新
  state_sem.release();
}


// remote call action
float[] callActionByRemote()
{
  try
  {
    // action_sem will wait, utill the server receive the action
    action_sem.acquire();
  }
  catch (Exception ex)
  {
    System.out.println(ex);
  }
  return action;
}


// Lilypad回传奖励函数，不需要
/*void reward(float COST)
{
  float target_reward = COST;
  if ((test.t +0.12) > EpisodeTime){
    done = true;
    dat.finish(); 
    simNum = simNum + 1;
    setUpNewSim(simNum);
  }
 
  println("time:" + test.t);
  //done = false;
  // TODO: !!! update reward , done and state in buffer
  reward_buffer = target_reward;
  //release state semaphore to let server return the resulted state
  state_sem.release();
}*/



// 旧的启动方式，不包含流场数据符号
/*WebServer StartServer(int port)
{
  println(port);
  WebServer server = new WebServer(port);
  server.addHandler("connect", new serverHandler());
  server.start();

  System.out.println("Started server successfully.");
  System.out.println("Accepting requests. (Halt program to stop.)");
  return server;
}*/


WebServer StartServer(int port, boolean includeFlow) {
    println(port);
    WebServer server = new WebServer(port);
    server.addHandler("connect", new serverHandler(includeFlow));
    server.start();

    System.out.println("Started server successfully.");
    System.out.println("Accepting requests. (Halt program to stop.)");
    return server;
}


// server handler to provide api
public class serverHandler {
    
    // 成员变量存储 includeFlow 标志
    private boolean includeFlow; 

    // 构造函数,启用一个新的serverHandler类时包含流场标识
    public serverHandler(boolean includeFlow) {
        this.includeFlow = includeFlow; // 初始化标志
    }

    public String Step(String actionInJson) {
        JSONObject output_object = new JSONObject();
        // 解析传入的动作数据
        JSONObject input_object = JSONObject.parseObject(actionInJson);

        // 更新动作数据
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");
        action[2] = input_object.getFloat("v3");

        // 释放动作信号量
        action_sem.release();

        // 等待状态信号量以获取更新后的状态
        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            println(e);
            println("[Error] state do not refresh");
        }

        // 构造输出数据，仅返回状态
        output_object.put("state", state);
        // 如果 includeFlow 为 true，添加流场数据
        if (includeFlow) {
            output_object.put("flow_u", flowVelocityX);
            output_object.put("flow_v", flowVelocityY);
        }
        //output_object.put("flow_u", flowVelocityX);
        //output_object.put("flow_v", flowVelocityY);

        return output_object.toJSONString();
    }

    public String query_state() {
        JSONObject output_object = new JSONObject();
        // 仅返回当前状态
        output_object.put("state", state);
        // 如果 includeFlow 为 true，添加流场数据
        if (includeFlow) {
            output_object.put("flow_u", flowVelocityX);
            output_object.put("flow_v", flowVelocityY);
        }
        //output_object.put("flow_u", flowVelocityX);
        //output_object.put("flow_v", flowVelocityY);
        return output_object.toJSONString();
    }

    public String reset(String actionInJson) {
        JSONObject input_object = JSONObject.parseObject(actionInJson);
        JSONObject output_object = new JSONObject();

        // 重置动作数据
        action[0] = input_object.getFloat("v1");
        action[1] = input_object.getFloat("v2");
        action[2] = input_object.getFloat("v3");

        println("Reset_simNum " + simNum);
        setUpNewSim(simNum);
        simNum += 1;// 每次 reset 仿真编号递增

        // 释放动作信号量
        action_sem.release();
        println("action released");

        // 等待状态信号量以获取更新后的状态
        try {
            state_sem.acquire();
        } catch (InterruptedException e) {
            print(e);
        }

        // 构造输出数据，仅返回状态
        output_object.put("state", state);
        // 如果 includeFlow 为 true，添加流场数据
        if (includeFlow) {
            output_object.put("flow_u", flowVelocityX);
            output_object.put("flow_v", flowVelocityY);
        }
        //output_object.put("flow_u", flowVelocityX);
        //output_object.put("flow_v", flowVelocityY);
        println("complete reset");
        return output_object.toJSONString();
    }
}


public String multy_state(PVector pos, PVector vel, ArrayList<Float> surfacePressures) {
  JSONObject multy_state_json = new JSONObject();
  multy_state_json.put("vel_x", vel.x);
  multy_state_json.put("vel_y", vel.y);
  multy_state_json.put("vel_angle", vel.z);
  multy_state_json.put("pos_x", pos.x);
  multy_state_json.put("pos_y", pos.y);
  multy_state_json.put("angle", pos.z);
  multy_state_json.put("surfacePressures_1", surfacePressures.get(0));
  multy_state_json.put("surfacePressures_2", surfacePressures.get(1));
  multy_state_json.put("surfacePressures_3", surfacePressures.get(2));
  multy_state_json.put("surfacePressures_4", surfacePressures.get(3));
  multy_state_json.put("surfacePressures_5", surfacePressures.get(4));
  multy_state_json.put("surfacePressures_6", surfacePressures.get(5));
  multy_state_json.put("surfacePressures_7", surfacePressures.get(6));
  multy_state_json.put("surfacePressures_8", surfacePressures.get(7));
 return multy_state_json.toJSONString();
}


/*public String flowFieldToJson(VectorField flowVelocity) {
    JSONObject flowField = new JSONObject();
    JSONArray xField = new JSONArray();
    JSONArray yField = new JSONArray();

    for (int i = 0; i < flowVelocity.x.a.length; i++) {
        JSONArray xRow = new JSONArray();
        JSONArray yRow = new JSONArray();
        for (int j = 0; j < flowVelocity.x.a[i].length; j++) {
            xRow.add(flowVelocity.x.a[i][j]);
            yRow.add(flowVelocity.y.a[i][j]);
        }
        xField.add(xRow);
        yField.add(yRow);
    }

    flowField.put("x", xField);
    flowField.put("y", yField);
    return flowField.toJSONString();
}*/



void setUpNewSim(int runNum){       
  int xLengths = 20, yLengths = 8, zoom = 100/resolution, Re = 500;    
  float gR = 3;        
  float xi0 = 0, xi1 = 0, xi2 = 0, theta = PI/6;  
  
  smooth();
  
  if (zoom <= 1){zoom = 1;}
  
  test = new Pinball(resolution, Re, gR, theta, xi0, xi1, xi2, tStep, xLengths, yLengths, false);          
  dat = new SaveScalar("saved/"+str(runNum)+".txt", (float)resolution, (float)xLengths, (float)yLengths, 32);      
  
  new File(datapath + str(runNum)).mkdir();
  nextactionA=0;
  nextactionB=0;
  nextactionC=0;
}