import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

// Register once for the whole app.
gsap.registerPlugin(ScrollTrigger, useGSAP);
gsap.defaults({ ease: "expo.out", duration: 1 });

export { gsap, ScrollTrigger, useGSAP };
