#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "peseudo_random.hpp"
#include "randomsampler.hpp"
#include "online.hpp"

namespace cylinder_reflectance {
	inline double normalized_gaussian(double beta, double theta) {
		return std::exp(-theta * theta / (2.0 * beta * beta)) / (std::sqrt(glm::two_pi<double>()) * beta);
	}
	inline double Nr(double phi_d) {
		return 0.25 * std::cos(phi_d * 0.5);
	}
	inline double Mr(double theta_h, double gamma_s) {
		return normalized_gaussian(gamma_s, theta_h);
	}
	inline double bsdf(glm::vec3 u, glm::vec3 wi, glm::vec3 wo, double beta, double *cosThetaI) {
		double sinThetaI = glm::dot(wi, u);
		double thetaI = std::asin(sinThetaI);
		glm::dvec3 wi_on_normal = wi - u * sinThetaI;

		double sinThetaO = glm::dot(wo, u);
		double thetaO = std::asin(sinThetaO);
		glm::dvec3 wo_on_normal = wo - u * sinThetaO;

		double phi_d = std::acos(glm::dot(wi_on_normal, wo_on_normal) / (glm::length(wi_on_normal) * glm::length(wo_on_normal)));
		double theta_h = (thetaI + thetaO) * 0.5;
		double theta_d = (thetaI - thetaO) * 0.5;

		*cosThetaI = std::cos(thetaI);

		return Nr(phi_d) * Mr(theta_h, beta) / std::pow(cos(theta_d), 2);
	}

	inline void plot_cylinder_reflectance() {
		glm::dvec3 u = glm::dvec3(0, 0, 1);

		for (float beta : {glm::radians(1.0), glm::radians(5.0), glm::radians(10.0), glm::radians(25.0)}) {
			printf("beta = %.5f (%.2f deg)\n", beta, glm::degrees(beta));

			for (float oTheta = 0.0; oTheta < glm::pi<double>() * 0.5; oTheta += 0.02) {
				/*
				[Point Wrangle]
					float theta = ((float)@ptnum / (float)npoints(0)) * radians(90);
					@P.z = sin(theta);
					@P.y = cos(theta);
					@P.x = 0;
				*/
				glm::dvec3 wo = glm::dvec3(0, std::cos(oTheta), std::sin(oTheta));

				rt::OnlineMean<double> albedo;
				rt::XoroshiroPlus128 random;
				for (int i = 0; i < 2000000; ++i) {
					double p = 1.0 / (glm::pi<double>() * 4.0);
					glm::dvec3 wi = rt::sample_on_unit_sphere(&random);

					double cosThetaI;
					double fs = bsdf(u, wi, wo, beta, &cosThetaI);
					double value = fs * cosThetaI / p;

					if (value < 0.0) { abort(); }

					albedo.addSample(value);
				}
				printf("%.5f\n", albedo.mean());
			}
			printf("---\n");
		}
	}
}