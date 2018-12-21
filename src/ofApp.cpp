#include "ofApp.h"
#include "ofxImGuiLite.hpp"
#include "peseudo_random.hpp"
#include "randomsampler.hpp"
#include "online.hpp"
#include "cylinder_reflectance.hpp"

#if NDEBUG
#define RT_ASSERT(expect_true) ;
#else
#define RT_ASSERT(expect_true) if((expect_true) == 0) { __debugbreak(); }
#endif

inline double normalized_gaussian(double beta, double theta) {
	return std::exp(-theta * theta / (2.0 * beta * beta)) / (std::sqrt(glm::two_pi<double>() * beta * beta));
}

// ex)
// air to glass
// eta_t(glass) = 1.5, eta_i(air) = 1.0
inline double fresnel_dielectrics(double cosTheta, double eta_t, double eta_i) {
	auto sqr = [](double x) { return x * x; };

	double c = cosTheta;
	double g = std::sqrt(sqr(eta_t) / sqr(eta_i) - 1.0 + sqr(c));

	double a = 0.5 * sqr(g - c) / sqr(g + c);
	double b = 1.0 + sqr(c * (g + c) - 1.0) / sqr(c * (g - c) + 1.0);
	return a * b;
}

inline double fr_cosTheta(double theta_d, double phi_d) {
	return std::cos(theta_d) * std::cos(phi_d * 0.5);
}

inline double scattering_rs(double phi_d, double theta_h, double gamma_s) {
	return std::cos(phi_d * 0.5) * normalized_gaussian(gamma_s, theta_h);
}
inline double scattering_rv(double theta_h, double gamma_v, double kd, double cosThetaI, double cosThetaO) {
	return ((1.0 - kd) * normalized_gaussian(gamma_v, theta_h) + kd) / (cosThetaI + cosThetaO);
}

inline glm::dvec3 bsdf(double theta_d, double theta_h, double phi_d, double cosThetaI, double cosThetaO, double gamma_s, double gamma_v, double kd, double eta_t, glm::dvec3 A) {
	double eta_i = 1.0;
	RT_ASSERT(1.0 <= eta_i);

	// f_r,s
	// return glm::dvec3(1.0) * scattering_rs(phi_d, theta_h, gamma_s) / std::pow(cos(theta_d), 2);

	double Fr_cosTheta_i = fr_cosTheta(theta_d, phi_d);
	auto safeSqrt = [](double x) {
		return std::sqrt(std::max(x, 0.0));
	};

	double Fr = fresnel_dielectrics(Fr_cosTheta_i, eta_t, eta_i);
	double Ft = (1.0 - Fr);
	double F = Ft * Ft;

	double rs = Fr * scattering_rs(phi_d, theta_h, gamma_s);
	double rv = F * scattering_rv(theta_h, gamma_v, kd, cosThetaI, cosThetaO);

	// f_r,v
	// return (scattering_rv(theta_h, gamma_v, kd, *cosThetaI, cosThetaO) * A) / std::pow(cos(theta_d), 2);

	return (glm::dvec3(rs) + rv * A) / std::pow(cos(theta_d), 2);
}

struct ThreadGeometricParameters {
	double theta_d;
	double theta_h;
	double phi_d;
	double cosPhiI;
	double cosPhiO;
	double cosThetaI;
	double cosThetaO;
	double psi_d;
	double cosPsiI;
	double cosPsiO;
};

inline bool parameterize(glm::dvec3 u, glm::dvec3 v, glm::dvec3 n, glm::dvec3 wi, glm::dvec3 wo, ThreadGeometricParameters *params) {
	RT_ASSERT(std::abs(glm::length2(u) - 1.0) < 1.0e-4);
	RT_ASSERT(std::abs(glm::length2(n) - 1.0) < 1.0e-4);
	RT_ASSERT(std::abs(glm::length2(wi) - 1.0) < 1.0e-4);
	RT_ASSERT(std::abs(glm::length2(wo) - 1.0) < 1.0e-4);

	double sinThetaI = glm::clamp(glm::dot(wi, u), -1.0, 1.0); // clamp for asin stability
	double thetaI = std::asin(sinThetaI);
	double sinThetaO = glm::clamp(glm::dot(wo, u), -1.0, 1.0); // clamp for asin stability
	double thetaO = std::asin(sinThetaO);

	glm::dvec3 wi_on_normal = glm::normalize(wi - u * sinThetaI);
	glm::dvec3 wo_on_normal = glm::normalize(wo - u * sinThetaO);

	// Φが定義できない
	if (glm::all(glm::isfinite(wi_on_normal)) == false) {
		return false;
	}
	if (glm::all(glm::isfinite(wo_on_normal)) == false) {
		return false;
	}

	double cosPhiD = glm::clamp(glm::dot(wi_on_normal, wo_on_normal), -1.0, 1.0); // clamp for acos stability

	double phi_d = std::acos(cosPhiD);
	double theta_h = (thetaI + thetaO) * 0.5;
	double theta_d = (thetaI - thetaO) * 0.5;

	double cosThetaO = std::cos(thetaO);

	glm::dvec3 wi_on_tangent_normal = glm::normalize(wi - v * glm::dot(wi, v));
	glm::dvec3 wo_on_tangent_normal = glm::normalize(wo - v * glm::dot(wo, v));

	// ψが定義できない
	if (glm::all(glm::isfinite(wi_on_tangent_normal)) == false) {
		return false;
	}
	if (glm::all(glm::isfinite(wo_on_tangent_normal)) == false) {
		return false;
	}
	double cosPsiD = glm::clamp(glm::dot(wi_on_tangent_normal, wo_on_tangent_normal), -1.0, 1.0); // clamp for acos stability
	double psi_d = std::acos(cosPsiD);

	params->theta_d = theta_d;
	params->theta_h = theta_h;
	params->cosPhiI = glm::dot(n, wi_on_normal);
	params->cosPhiO = glm::dot(n, wo_on_normal);
	params->phi_d = phi_d;
	params->cosThetaI = std::cos(thetaI);
	params->cosThetaO = std::cos(thetaO);
	params->psi_d = psi_d;
	params->cosPsiI = glm::dot(n, wi_on_tangent_normal);
	params->cosPsiO = glm::dot(n, wo_on_tangent_normal);

	return true;
}

inline glm::dvec3 bsdf(glm::dvec3 u, glm::dvec3 v, glm::dvec3 n, glm::dvec3 wi, glm::dvec3 wo, double gamma_s, double gamma_v, double kd, double eta_t, glm::dvec3 A, double *cosThetaI) {
	RT_ASSERT(0.0 <= gamma_s);
	RT_ASSERT(0.0 <= gamma_v);
	RT_ASSERT(1.0 <= eta_t);
	RT_ASSERT(0.0 <= kd && kd <= 1.0);

	ThreadGeometricParameters p;
	if (parameterize(u, v, n, wi, wo, &p) == false) {
		return glm::dvec3(0.0);
	}

	*cosThetaI = p.cosThetaI;
	RT_ASSERT(0.0 <= *cosThetaI);

	return bsdf(p.theta_d, p.theta_h, p.phi_d, p.cosThetaI, p.cosThetaO, gamma_s, gamma_v, kd, eta_t, A);
}

// inline glm::dvec3 bsdf

inline glm::dvec3 polar_to_cartesian(double theta, double phi) {
	double sinTheta = std::sin(theta);
	glm::dvec3 v = {
		sinTheta * std::cos(phi),
		sinTheta * std::sin(phi),
		std::cos(theta)
	};
	return v;
};

//--------------------------------------------------------------
void ofApp::setup() {
	ofxImGuiLite::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(10.0f);
	_camera.setPosition(glm::length(-_camera.getPosition()), 0, 0);
	_camera.lookAt(glm::vec3(0, 0, 0));

	// cylinder_reflectance::plot_cylinder_reflectance();
}
void ofApp::exit() {
	ofxImGuiLite::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw(){
	//static float gamma_s = glm::radians(5.0);
	//static float gamma_v = glm::radians(5.0);
	//static float eta = 1.5;
	//static float kd = 0.2;
	//static glm::vec3 A = glm::vec3(1.0);

	// (c) Flat
	//static float gamma_s = glm::radians(2.5);
	//static float gamma_v = glm::radians(5.0);
	//static float eta = 1.539;
	//static float kd = 0.1;
	//static glm::vec3 A = glm::vec3(1.0, 0.37, 0.3) * 0.035;

	// (c) Twisted
	static float gamma_s = glm::radians(30.0);
	static float gamma_v = glm::radians(60.0);
	static float eta = 1.539;
	static float kd = 0.7;
	static glm::vec3 A = glm::vec3(1.0, 0.37, 0.3) * 0.2;

	// (b) Twisted
	//static float gamma_s = glm::radians(18.0);
	//static float gamma_v = glm::radians(32.0);
	//static float eta = 1.345;
	//static float kd = 0.3;
	//static glm::vec3 A = glm::vec3(1.0, 0.95, 0.05) * 0.16;

	static float theta_i = 0.0f;
	static float phi_i = glm::radians(90.0);
	static float theta_o = 0.0f;
	static float phi_o = 0.0f;


	ofEnableDepthTest();

	ofClear(0);

	_camera.begin();
	ofPushMatrix();
	//ofRotateZDeg(90.0f);
	ofSetColor(64);
	ofDrawGridPlane(1.0f);
	ofPopMatrix();

	ofDrawAxis(50);

	ofSetColor(255);

	// draw
	glm::dvec3 wi = polar_to_cartesian(theta_i, phi_i);
	glm::dvec3 wo = polar_to_cartesian(theta_o, phi_o);

	ofSetColor(255, 0, 0);
	ofDrawLine(glm::vec3(), wi);
	ofDrawBitmapString("wi", glm::vec3(wi));

	ofSetColor(0, 255, 0);
	ofDrawLine(glm::vec3(), wo);
	ofDrawBitmapString("wo", glm::vec3(wo));

	glm::dvec3 u = glm::dvec3(0, 0, 1);
	glm::dvec3 v = glm::dvec3(1, 0, 0);
	
	glm::dvec3 n = glm::dvec3(0, 1, 0);

	double sinThetaI = glm::dot(wi, u);
	double thetaI = std::asin(sinThetaI);
	glm::dvec3 wi_on_normal = wi - u * sinThetaI;


	glm::dvec3 wi_on_tangent_normal = wi - v * glm::dot(wi, v);
	ofSetColor(255);
	ofDrawLine(glm::vec3(), wi_on_tangent_normal);
	ofDrawBitmapString("wi_on_tangent_normal", glm::vec3(wi_on_tangent_normal));

	// ofMesh plot_rs = ofMesh::sphere(1.0f, 24, OF_PRIMITIVE_TRIANGLES);
	// ofMesh plot_rs = ofMesh::cylinder(1.0f, 0.01f, 500, 1, 1);
	//std::vector<glm::vec3> &plot_rs_vertices = plot_rs.getVertices();
	//for (int i = 0; i < plot_rs_vertices.size(); ++i) {
	//	glm::dvec3 wo = plot_rs_vertices[i];
	//	plot_rs_vertices[i] = glm::dvec3(wo.y, -wo.x, wo.z);
	//}

	 // https://www.wolframalpha.com/input/?i=Solve%5Bx%3Db*exp(a*y)+%2B+c,+y%5D
	 // https://www.desmos.com/calculator/e5xh41mgdp
	 auto log_scale = [](double x) {
		 double a = 1.01706;
		 double b = -3.74685;
		 double c = -0.023734;
		 return (std::log(x - c) - b) / a;
	 };

	// polar grid
	for (float r : {0.04f, 0.16f, 0.48f, 1.34f}) {
		char text[64];
		sprintf(text, "%.2f", r);

		ofPolyline poly;
		r = log_scale(r);
		poly.arc(0, 0, 0, r, r, 0, 360, 100);
		std::vector<glm::vec3> &plot_rs_vertices = poly.getVertices();
		for (int i = 0; i < plot_rs_vertices.size(); ++i) {
			glm::dvec3 wo = plot_rs_vertices[i];
			plot_rs_vertices[i] = glm::dvec3(wo.z, wo.y, -wo.x);
		}
		ofSetColor(64);
		poly.draw();

		ofSetColor(255);
		ofDrawBitmapString(std::string(text), glm::vec3(0, 0, r));
	}

	// 2D プロット
	ofPolyline poly[3];
	for (int j = 0; j < 3; ++j) {
		poly[j].arc(0, 0, 0, 1, 1, 0, 360, 1000);
		std::vector<glm::vec3> &plot_rs_vertices = poly[j].getVertices();
		for (int i = 0; i < plot_rs_vertices.size(); ++i) {
			glm::dvec3 wo = plot_rs_vertices[i];
			plot_rs_vertices[i] = glm::dvec3(wo.z, wo.y, -wo.x);
		}

		for (int i = 0; i < plot_rs_vertices.size(); ++i) {
			glm::dvec3 wo = plot_rs_vertices[i];

			double cosThetaO;
			glm::dvec3 fs = bsdf(u, v, n, wi, wo, gamma_s, gamma_v, kd, eta, A, &cosThetaO);
			plot_rs_vertices[i] *= log_scale(fs[j]);

			// double fs_avg = (fs.x + fs.y + fs.z) / 3.0;
			// plot_rs_vertices[i] *= log_scale(fs_avg);
		}
	}

	ofSetColor(255, 0, 0);
	poly[0].draw();
	ofSetColor(0, 255, 0);
	poly[1].draw();
	ofSetColor(0, 0, 255);
	poly[2].draw();

	// sphere
	//ofMesh plot_rs = ofMesh::sphere(1.0f, 24, OF_PRIMITIVE_TRIANGLES);
	//std::vector<glm::vec3> &plot_rs_vertices = plot_rs.getVertices();
	//for (int i = 0; i < plot_rs_vertices.size(); ++i) {
	//	glm::dvec3 wo = plot_rs_vertices[i];

	//	double cosThetaO;
	//	glm::dvec3 fs = bsdf(u, wi, wo, gamma_s, gamma_v, kd, eta, A, &cosThetaO);
	//	double avg = (fs.x + fs.y + fs.z) / 3.0;
	//	plot_rs_vertices[i] *= log_scale(avg);
	//}
	//ofSetColor(255);
	//plot_rs.drawWireframe();


	//ofSetColor(255);
	//poly.draw();
	// 

	//double sinThetaO = glm::dot(wo, u);
	//double thetaO = std::asin(sinThetaO);
	//glm::dvec3 wo_on_normal = wo - u * sinThetaO;
	//double phi_d = std::acos(glm::dot(wi_on_normal, wo_on_normal) / (glm::length(wi_on_normal) * glm::length(wo_on_normal)));

	//double theta_d = (thetaI - thetaO) * 0.5;

	//glm::dvec3 wh = glm::normalize(wo + wi);
	//ofSetColor(255);
	//ofDrawLine(glm::vec3(), wh);
	//ofDrawBitmapString("wh", glm::vec3(wh));

	//double f_radian_0 = std::acos(glm::dot(wi, wh));
	//double f_radian_1 = std::acos(std::cos(theta_d) * std::cos(phi_d * 0.5));

	//ofSetColor(255);
	//ofDrawLine(glm::vec3(), wi_on_normal);
	//ofSetColor(255);
	//ofDrawLine(glm::vec3(), wo_on_normal);

	// フレネルテスト
	//ofPolyline line;
	//for (int i = 0; i < 1000; ++i) {
	//	double theta = ofMap(i, 0, 1000, 0, glm::pi<float>() * 0.5);
	//	line.addVertex(theta, fresnel_dielectrics(cos(theta), 1.5, 1.0), 0);
	//}
	//ofSetColor(255);
	//line.draw();

	ofPushMatrix();
	ofRotateX(90);
	ofSetColor(128);
	ofDrawCylinder(0.004, 2.0f);
	ofPopMatrix();

	_camera.end();

	ofDisableDepthTest();
	ofSetColor(255);

	ofxImGuiLite::ScopedImGui imgui;

	// camera control                                          for control clicked problem
	if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || (ImGui::IsAnyWindowFocused() && ImGui::IsAnyMouseDown())) {
		_camera.disableMouseInput();
	}
	else {
		_camera.enableMouseInput();
	}

	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiSetCond_Appearing);
	ImGui::SetNextWindowSize(ImVec2(260, 330), ImGuiSetCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiSetCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);

	if (ImGui::Button("Look from -X")) {
		_camera.setPosition(glm::length(-_camera.getPosition()), 0, 0);
		_camera.lookAt(glm::vec3(0, 0, 0));
	}
	if (ImGui::Button("Look from Z")) {
		_camera.setPosition(0, 0, glm::length(_camera.getPosition()));
		_camera.lookAt(glm::vec3(0, 0, 0));
	}

	ImGui::SliderFloat("gamma_s", &gamma_s, 0.0, 1.5f);
	ImGui::SliderFloat("gamma_v", &gamma_v, 0.0, 1.5f);
	ImGui::SliderFloat("eta", &eta, 1.0f, 2.0f);
	ImGui::SliderFloat("kd", &kd, 0.0f, 1.0f);
	ImGui::InputFloat3("A", glm::value_ptr(A));
	
	ImGui::SliderFloat("theta_i", &theta_i, 0.0, glm::pi<float>());
	ImGui::SliderFloat("phi_i (wi)", &phi_i, -glm::pi<float>(), glm::pi<float>());

	//if (ImGui::Button("theta_i = 15 deg")) {
	//	theta_i = glm::radians(90.0) + glm::radians(15.0);
	//}
	//if (ImGui::Button("theta_i = 30 deg")) {
	//	theta_i = glm::radians(90.0) + glm::radians(30.0);
	//}
	//if (ImGui::Button("theta_i = 45 deg")) {
	//	theta_i = glm::radians(90.0) + glm::radians(45.0);
	//}
	//ImGui::SliderFloat("theta_o", &theta_o, 0.0, glm::pi<float>());
	//ImGui::SliderFloat("phi_o (wo)", &phi_o, -glm::pi<float>(), glm::pi<float>());
	
	//float phi_i_shade = std::atan2(glm::dot(v, wi),glm::dot(n, wi));
	//float phi_o_shade = std::atan2(glm::dot(v, wo), glm::dot(n, wo));
	//ImGui::Text("phi_o %.3f", glm::degrees(phi_o_shade));
	//ImGui::Text("phi_i %.3f", glm::degrees(phi_i_shade));

	//float phi_d_shade = phi_i_shade - phi_o_shade;
	//ImGui::Text("phi_d_shade %.3f", glm::degrees(phi_d_shade));

	//ImGui::Text("thetaI %.3f", glm::degrees(thetaI));
	//ImGui::Text("thetaO %.3f", glm::degrees(thetaO));
	//ImGui::Text("phi_d %.3f", glm::degrees(phi_d));

	// ImGui::Text("albedo %.3f", albedo.mean());
	// ImGui::Text("f_radian_0 %.3f", f_radian_0);
	// ImGui::Text("f_radian_1 %.3f", f_radian_1);
	
	ImGui::End();

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
