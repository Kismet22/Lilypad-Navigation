class Pinball {
  BDIM flow;
  BodyUnion body;
  boolean QUICK = true, order2 = true;
  int n, m, out, up, resolution,NT=1;
  float dt, t, D, xi0, xi1, xi2,theta;
  float xi0_m, xi1_m, xi2_m, gR, theta_m, r, dphi0, dphi1, dphi2;
  float pos_x_default, pos_y_default;
  FloodPlot flood;
  PVector force, force_0, force_1, force_2;
  PVector vel, pos;
  ArrayList<Float> surfacePressures;

  Pinball (int resolution, int Re,  float gR,  float theta,  float xi0,  float xi1,  float xi2, float dtReal, int xLengths, int yLengths, boolean isResume, float _pos_x, float _pos_y) {
    // resolution:理解成缩放倍数
    // resolution取16
    n = xLengths*resolution;
    m = yLengths*resolution;
    if (_pos_x != 0.0f || _pos_y != 0.0f) {
        // 不同时为0，使用传入坐标
        pos_x_default = _pos_x;
        pos_y_default = _pos_y;
    } else {
        // 否则使用默认坐标(240,64)
        pos_x_default = 6 * n / 8;  
        pos_y_default = m / 2;  
    }

    this.resolution = resolution;
    this.xi0 = xi0;
    this.xi1 = xi1;
    this.xi2 = xi2;
    this.gR = gR;
    this.theta = theta;
    this.dt = dtReal*this.resolution;
    theta_m=theta;

    Window view = new Window(0, 0, n, m); // zoom the display around the body
    D=resolution;

    float r=D+gR*D;
    body =new BodyUnion(new CircleBody(n/6, m/2+r/2, D, view),
    new CircleBody(n/6, m/2-r/2, D, view),
    new CircleBody(n/6+r*cos(theta), m/2, D, view),
    // 椭圆体的坐标,定义在Body.pde文件中
    new EllipseBody(pos_x_default, pos_y_default, D/2, 1.5, view));
    flow = new BDIM(n,m,dt,body,(float)D/Re,QUICK);
    
    if(isResume){
      // flow.resume("saved_1/init/init.bdim");// initial state with swimmer
      flow.resume("saved_1/init/init_1.bdim");// initial state without swimmer
    }
    
    flood = new FloodPlot(view);
    flood.range = new Scale(-1, 1);
    flood.setLegend("vorticity"); 

    force_0 = new PVector(); // 初始化 force_0
  }


void update2(){
  //dt = flow.checkCFL();
  flow.dt = dt;
  force_0.x = xi0;
  force_0.y = xi1;
  dphi0 = xi2; // torque, 顺时针为正

  // body.bodyList.get(3).react(flow);
  body.bodyList.get(3).react(flow,force_0,dphi0);
              
  flow.update(body);
  if (order2) {flow.update2(body);}

  t += dt/resolution;  //nonedimension
  
  vel = body.bodyList.get(3).dotxc;
  vel.z = body.bodyList.get(3).dotphi;
  pos = body.bodyList.get(3).xc;
  pos.z = body.bodyList.get(3).phi;
  // 计算表面压力
  // 处理 surfacePressures 列表中的数据
  surfacePressures = body.bodyList.get(3).calculateSurfacePressures(flow.p);

  //flow.u中存储速度信息
  //println("Flow velocity at target [88][64]: " + flow.u.x.a[88][64] + ", " + flow.u.y.a[88][64]);
 }

  void display(float targetx, float targety) {
    flood.display(flow.u.curl());
    body.display();
    flood.displayTime(t);
    
    
    // float xPos_target = 140; 
    // float yPos_target = 55; 
    float diameter_target = 8; 
    float diameter = 3; 
    
    fill(255, 0, 0); 
    ellipse(targetx, targety, diameter_target, diameter_target); 
  }
}
