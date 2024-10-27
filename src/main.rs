#![allow(dead_code)]
use core::f32;
use std::time::Instant;

use image::{ImageBuffer, Rgb};
use glam::{swizzles::*, Vec2, Vec3, Vec3A, Vec4};
use rayon::prelude::*;

fn main() {
    let imgx = 1920 / 2;
    let imgy = 1080 / 2;

    let samples_per_pixel = 250;
    let max_bounces = 10;

    let vertical_fov = 20.;
    let look_from = Vec3A::new(13.0, 2.0, 3.0);
    let look_at = Vec3A::new(0.0, 0.0, 0.0);
    let vup = Vec3A::new(0.0, 1.0, 0.0);

    let defocus_angle = 0.6;
    let focus_dist = 10.0;

    let camera = Camera::new(
        imgx,
        imgy,
        vertical_fov,
        look_from,
        look_at,
        vup,
        defocus_angle,
        focus_dist,
    );

    // world

    let mut spheres = vec![];
    let mut materials = vec![];

    for z in -11..11 {
        for x in -11..11 {
            let choice = rand::random();
            let center = Vec3A::new(x as f32 + 0.9 * rand::random::<f32>(), 0.2, z as f32 + 0.9 * rand::random::<f32>());
            let mut center2 = None;

            if (center - Vec3A::new(4.0, 0.2, 0.0)).length_squared() > (0.9 * 0.9) {
                let material = match choice {
                    0.0..0.8 => {
                        let albedo = Vec3A::from(rand::random::<[f32; 3]>()) * Vec3A::from(rand::random::<[f32; 3]>());
                        center2 = Some(center + Vec3A::new(0.0, 0.5 * rand::random::<f32>(), 0.0));
                        Mat::Lambertian(Lambertian { albedo })
                    }
                    0.8..0.95 => {
                        let albedo = (Vec3A::from(rand::random::<[f32; 3]>()) + Vec3A::ONE) / 2.0;
                        let fuzz = rand::random::<f32>() / 2.0;
                        Mat::Metal(Metal { albedo, fuzz })
                    }
                    _ => {
                        Mat::Dielectric(Dielectric { refraction_index: 1.5 })
                    }
                };
                materials.push(material);
                let material_index = materials.len() - 1;

                let sphere = if let Some(c2) = center2 {
                    Sphere::new_moving(center, c2, 0.2, material_index as u32)
                } else {
                    Sphere::new(center, 0.2, material_index as u32)
                };
                spheres.push(sphere);
            }
        }
    }

    materials.push(Mat::Dielectric(Dielectric { refraction_index: 1.5 }));
    spheres.push(Sphere::new(Vec3A::new(0.0, 1.0, 0.0), 1.0, materials.len() as u32 - 1));

    materials.push(Mat::Lambertian(Lambertian { albedo: col(0.4, 0.2, 0.1) }));
    spheres.push(Sphere::new(Vec3A::new(-4.0, 1.0, 0.0), 1.0, materials.len() as u32 - 1));

    materials.push(Mat::Metal(Metal { albedo: col(0.7, 0.6, 0.5), fuzz: 0.0 }));
    spheres.push(Sphere::new(Vec3A::new(4.0, 1.0, 0.0), 1.0, materials.len() as u32 - 1));

    materials.push(Mat::Lambertian(Lambertian { albedo: col(0.5, 0.5, 0.5) }));
    spheres.push(Sphere::new(Vec3A::new(0.0, -1000.0, 0.0), 1000.0, materials.len() as u32 - 1));

    let spheres = HittableList::from_list(spheres);
    let world = World { spheres, materials };

    let mut imgbuf = ImageBuffer::new(imgx, imgy);

    let start_time = Instant::now();
    imgbuf.enumerate_pixels_mut().par_bridge().for_each(
        |(x, y, pixel)| {
            let mut colour = Vec3A::ZERO;
            for _ in 0..samples_per_pixel {
                let ray = camera.ray_at(x, y);
                colour += ray_colour(&world, ray, max_bounces);
            }
            write_colour(colour / samples_per_pixel as f32, pixel);
        }
    );
    println!("Rendered in {:#?}", start_time.elapsed());

    imgbuf.save("image.png").unwrap();
}

fn ray_colour(world: &World, r: Ray, depth: u32) -> Vec3A {
    if depth == 0 {
        return Vec3A::ZERO;
    }

    if let Some(rec) = world.hit(r, Interval::new(0.001, f32::INFINITY)) {
        if let Some((scattered, attenuation)) = world.materials[rec.mat_index as usize].scatter(r, rec) {
            return attenuation * ray_colour(world, scattered, depth-1);
        }
        return Vec3A::ZERO;
    }

    let unit_direction = r.direction.normalize();
    let a = 0.5 * (unit_direction.y + 1.0);
    (1.0 - a) * col(1.0, 1.0, 1.0) + a * col(0.5, 0.7, 1.0)
}

struct BVHNode {
    left: Box<dyn Hittable>,
    right: Box<dyn Hittable>,
    bbox: AABB,
}

impl BVHNode {
    // fn new(objects: Vec<Box<dyn Hittable>>) -> Self {
    //     let len = objects.len();
    //     Self::split(objects, 0, len)
    // }

    // fn split(objects: Vec<Box<dyn Hittable>>, start: usize, end: usize) -> Self {
    //     let axis = rand::random::<usize>() % 3;

    //     let comparator 
    // }

    // fn box_compare(a: Box<dyn Hittable>, b: Box<dyn Hittable>, axis: usize) -> bool {
    //     let ai = a.bounding_box().axis_interval(axis);
    //     let bi = b.bounding_box().
    // }
}

impl Hittable for BVHNode {
    fn hit(&self, r: Ray, mut rt: Interval) -> Option<HitRecord> {
        if !self.bbox.hit(r, rt) {
            return None;
        }

        let left = self.left.hit(r, rt);
        if let Some(rec) = left {
            rt = Interval::new(rt.min, rec.t());
            let right = self.right.hit(r, rt);
            Some(right.unwrap_or(rec))
        } else {
            self.right.hit(r, rt)
        }
    }

    fn bounding_box(&self) -> AABB {
        self.bbox
    }
}

#[derive(Clone, Copy)]
struct AABB {
    x: Interval,
    y: Interval,
    z: Interval,
}

impl AABB {
    const EMPTY: AABB = AABB { x: Interval::EMPTY, y: Interval::EMPTY, z: Interval::EMPTY };

    fn new(a: Vec3A, b: Vec3A) -> Self {
        let x = if a.x <= b.x { Interval::new(a.x, b.x) } else { Interval::new(b.x, a.x) };
        let y = if a.y <= b.y { Interval::new(a.y, b.y) } else { Interval::new(b.y, a.y) };
        let z = if a.z <= b.z { Interval::new(a.z, b.z) } else { Interval::new(b.z, a.z) };
        AABB { x, y, z }
    }

    fn from_intervals(x: Interval, y: Interval, z: Interval) -> Self {
        Self { x, y, z }
    }

    fn axis_interval(&self, n: usize) -> Interval {
        match n {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!()
        }
    }

    fn hit(&self, r: Ray, mut rt: Interval) -> bool {
        for axis in 0..3 {
            let ax = self.axis_interval(axis);
            let adinv = 1.0 / r.direction[axis];

            let t0 = (ax.min - r.origin[axis]) * adinv;
            let t1 = (ax.max - r.origin[axis]) * adinv;
            if t0 < t1 {
                if t0 > rt.min { rt.min = t0 }
                if t1 < rt.max { rt.max = t1 }
            } else {
                if t1 > rt.min { rt.min = t1 }
                if t0 < rt.max { rt.max = t0 }
            }

            if rt.max <= rt.min {
                return false;
            }
        }
        true
    }

    fn merge(self, other: AABB) -> AABB {
        AABB {
            x: self.x.join(other.x),
            y: self.y.join(other.y),
            z: self.z.join(other.z),
        }
    }
}


#[derive(Clone, Copy)]
struct Interval {
    min: f32,
    max: f32,
}

impl Interval {
    const EMPTY: Interval = Interval { min: f32::MAX, max: f32::MIN };

    fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    fn clamp(self, x: f32) -> f32 {
        if x < self.min { return self.min }
        if x > self.max { return self.max }
        x
    }

    fn expand(self, delta: f32) -> Self {
        let padding = 0.5 * delta;
        Self {
            min: self.min - padding,
            max: self.max + padding,
        }
    }

    fn contains(self, x: f32) -> bool {
        self.min <= x && x <= self.max
    }

    fn join(self, other: Interval) -> Interval {
        Interval {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

struct World {
    spheres: HittableList<Sphere>,
    materials: Vec<Mat>,
}

impl Hittable for World {
    fn hit(&self, r: Ray, interval: Interval) -> Option<HitRecord> {
        self.spheres.hit(r, interval)
    }
    
    fn bounding_box(&self) -> AABB {
        self.spheres.bounding_box()
    }
}

trait Material {
    fn scatter(&self, ray: Ray, rec: HitRecord) -> Option<(Ray, Vec3A)>;
}

enum Mat {
    Metal(Metal),
    Lambertian(Lambertian),
    Dielectric(Dielectric),
}

impl Material for Mat {
    fn scatter(&self, ray: Ray, rec: HitRecord) -> Option<(Ray, Vec3A)> {
        match self {
            Mat::Metal(metal) => metal.scatter(ray, rec),
            Mat::Lambertian(lambertian) => lambertian.scatter(ray, rec),
            Mat::Dielectric(dielectric) => dielectric.scatter(ray, rec),
        }
    }
}

struct Dielectric {
    refraction_index: f32,
}

impl Material for Dielectric {
    fn scatter(&self, ray: Ray, rec: HitRecord) -> Option<(Ray, Vec3A)> {
        let attenuation = Vec3A::ONE;
        let ri = if rec.front_face() {
            1.0 / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray.direction.normalize();

        let cos_θ = (-unit_direction).dot(rec.normal()).min(1.0);
        let sin_θ = (1.0 - cos_θ * cos_θ).sqrt();

        let cannot_refract = ri * sin_θ > 1.0;
        let direction = if cannot_refract || reflectance(cos_θ, ri) > rand::random() {
            reflect(unit_direction, rec.normal())
        } else {
            refract(unit_direction, rec.normal(), ri)
        };

        let scattered = Ray {
            origin: rec.point(),
            direction,
            time: ray.time,
        };
        Some((scattered, attenuation))
    }
}

struct Metal {
    albedo: Vec3A,
    fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, ray: Ray, rec: HitRecord) -> Option<(Ray, Vec3A)> {
        let reflected = reflect(ray.direction, rec.normal());
        let reflected = reflected.normalize() + (self.fuzz * random_unit_vector());
        let scattered = Ray {
            origin: rec.point(),
            direction: reflected,
            time: ray.time,
        };
        if scattered.direction.dot(rec.normal()) > 0. {
            Some((scattered, self.albedo))
        } else {
            None
        }
    }
}

struct Lambertian {
    albedo: Vec3A,
}

impl Material for Lambertian {
    fn scatter(&self, ray: Ray, rec: HitRecord) -> Option<(Ray, Vec3A)> {
        let mut scatter_direction = rec.normal() + random_unit_vector();
        if scatter_direction.length_squared() < 1e-6 {
            scatter_direction = rec.normal();
        }
        let scattered = Ray {
            origin: rec.point(),
            direction: scatter_direction,
            time: ray.time,
        };
        Some((scattered, self.albedo))
    }
}

struct HittableList<T: Hittable>(Vec<T>, AABB);

impl<T: Hittable> HittableList<T> {
    fn new() -> Self {
        Self(vec![], AABB::EMPTY)
    }

    fn add(&mut self, item: T) {
        self.1 = self.1.merge(item.bounding_box());
        self.0.push(item);
    }

    fn from_list(l: impl Into<Vec<T>>) -> Self {
        let list: Vec<T> = l.into();
        let bbox = list.iter().fold(AABB::EMPTY, |b, o| b.merge(o.bounding_box()));
        Self(list, bbox)
    }
}

impl<T: Hittable> Hittable for HittableList<T> {
    fn hit(&self, r: Ray, interval: Interval) -> Option<HitRecord> {
        let mut best_hit = None;
        let mut closest_so_far = interval.max;

        for object in self.0.iter() {
            if let Some(rec) = object.hit(r, Interval::new(interval.min, closest_so_far)) {
                best_hit = Some(rec);
                closest_so_far = rec.t();
            }
        }

        best_hit
    }
    
    fn bounding_box(&self) -> AABB {
        self.1
    }
}

trait Hittable {
    fn hit(&self, r: Ray, interval: Interval) -> Option<HitRecord>;
    fn bounding_box(&self) -> AABB;
}

#[derive(Clone, Copy)]
struct HitRecord {
    normal_f: Vec4,
    point_t: Vec4,
    mat_index: u32,
}

impl HitRecord {
    fn new(ray: Ray, normal: Vec3A, point: Vec3A, t: f32, mat_index: u32) -> Self {
        let dot = ray.direction.dot(normal);
        let normal_f = if dot < 0.0 {
            normal.extend(dot)
        } else {
            (-normal).extend(dot)
        };
        let point_t = point.extend(t);
        Self {
            normal_f,
            point_t,
            mat_index,
        }
    }

    #[inline]
    fn t(self) -> f32 {
        self.point_t.w
    }

    #[inline]
    fn point(self) -> Vec3A {
        self.point_t.xyz().into()
    }

    #[inline]
    fn normal(self) -> Vec3A {
        self.normal_f.xyz().into()
    }

    #[inline]
    fn front_face(self) -> bool {
        self.normal_f.w < 0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Sphere(Vec4, Vec4);

impl Sphere {
    fn new(center: Vec3A, radius: f32, mat_index: u32) -> Self {
        let fmat = unsafe { std::mem::transmute(mat_index) };
        Self(center.extend(radius), Vec3A::ZERO.extend(fmat))
    }

    fn new_moving(c1: Vec3A, c2: Vec3A, radius: f32, mat_index: u32) -> Self {
        let fmat = unsafe { std::mem::transmute(mat_index) };
        Self(c1.extend(radius), (c2-c1).extend(fmat))
    }

    fn mat_index(&self) -> u32 {
        unsafe { std::mem::transmute(self.1.w) }
    }

    fn at(&self, time: f32) -> Vec3A {
        Vec3A::from(self.0.xyz()) + time * Vec3A::from(self.1.xyz())
    }

    #[inline(always)]
    fn radius(self) -> f32 {
        self.0.w
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: Ray, interval: Interval) -> Option<HitRecord> {
        let center: Vec3A = self.at(r.time);
        let radius = self.0.w;
        let oc = center - r.origin;
        let a = r.direction.length_squared();
        let h = r.direction.dot(oc);
        let c = oc.length_squared() - radius * radius;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();
        let mut root = (h - sqrt_d) / a;
        if !interval.contains(root) {
            root = (h + sqrt_d) / a;
            if !interval.contains(root) {
                return None;
            }
        }

        let point = r.at(root);
        let normal = (point - center) / radius;

        let rec = HitRecord::new(r, normal, point, root, self.mat_index());
        
        Some(rec)
    }

    fn bounding_box(&self) -> AABB {
        let rvec = Vec3A::splat(self.radius());
        if self.1.xyz() == Vec3::ZERO {
            let center = Vec3A::from(self.0.xyz());
            AABB::new(center - rvec, center + rvec)
        } else {
            let c1 = self.at(0.0);
            let box1 = AABB::new(c1 - rvec, c1 + rvec);
            let c2 = self.at(1.0);
            let box2 = AABB::new(c2 - rvec, c2 + rvec);
            box1.merge(box2)
        }

    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Ray {
    origin: Vec3A,
    direction: Vec3A,
    time: f32,
}

impl Ray {
    fn at(self, t: f32) -> Vec3A {
        self.origin + t * self.direction
    }

    fn new(origin: Vec3A, direction: Vec3A) -> Self {
        Self { origin, direction, time: 0.0 }
    }

    fn new_t(origin: Vec3A, direction: Vec3A, time: f32) -> Self {
        Self { origin, direction, time }
    }
}

#[derive(Debug)]
struct Camera {
    p0_loc: Vec3A,
    delta_u: Vec3A,
    delta_v: Vec3A,
    center: Vec3A,
    defocus_disk_u: Vec3A,
    defocus_disk_v: Vec3A,
    defocus_angle: f32,
}

impl Camera {
    fn new(
        image_width: u32,
        image_height: u32,
        vertical_fov: f32,

        look_from: Vec3A,
        look_at: Vec3A,
        vup: Vec3A,

        defocus_angle: f32,
        focus_dist: f32,
    ) -> Self {
        let center = look_from;

        let θ = vertical_fov.to_radians();
        let h = (θ/2.).tan();

        let viewport_height = 2. * h * focus_dist;
        let viewport_width = viewport_height * (image_width as f32 / image_height as f32);

        let w = (look_from - look_at).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;

        let delta_u = viewport_u / image_width as f32;
        let delta_v = viewport_v / image_height as f32;
        let top_left = center - focus_dist * w - viewport_u/2. - viewport_v/2.;
        let p0_loc = top_left + 0.5 * (delta_u + delta_v);
        let defocus_radius = focus_dist * (defocus_angle / 2.0).to_radians().tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;
        let result = Self { p0_loc, delta_u, delta_v, center, defocus_disk_u, defocus_disk_v, defocus_angle };
        result
    }

    fn ray_at(&self, x: u32, y: u32) -> Ray {
        let offset = Vec2::from(rand::random::<[f32; 2]>());
        let sample = self.p0_loc
            + (x as f32 + offset.x) * self.delta_u
            + (y as f32 + offset.y) * self.delta_v;
        let origin = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.defocus_disk_sample()
        };

        let direction = sample - origin;
        let time = rand::random();
        Ray { origin, direction, time }
    }

    fn defocus_disk_sample(&self) -> Vec3A {
        let p = random_in_unit_disc();
        self.center + (p.x * self.defocus_disk_u) + (p.y * self.defocus_disk_v)
    }
}

#[inline]
fn write_colour(colour: Vec3A, pixel: &mut Rgb<u8>) {
    // colour is [0..=1]
    let clamped = 255.999 * colour
        .map(linear_to_gamma)
        .clamp(Vec3A::ZERO, Vec3A::ONE);
    let arr = clamped.to_array();
    let rgb = Rgb([arr[0] as u8, arr[1] as u8, arr[2] as u8]);
    *pixel = rgb;
}

fn col(r: f32, g: f32, b: f32) -> Vec3A {
    Vec3A::new(r, g, b)
}

fn random_unit_vector() -> Vec3A {
    loop {
        let p = 2.0 * Vec3A::from(rand::random::<[f32; 3]>()) - Vec3A::ONE;
        let len_sqr = p.length_squared();
        if 1e-100 < len_sqr && len_sqr <= 1.0 {
            return p / len_sqr.sqrt()
        }
    }
}

fn random_on_hemisphere(normal: Vec3A) -> Vec3A {
    let unit_sphere = random_unit_vector();
    if unit_sphere.dot(normal) > 0. {
        unit_sphere
    } else {
        -unit_sphere
    }
}

#[inline(always)]
fn linear_to_gamma(linear_component: f32) -> f32 {
    if linear_component > 0. {
        linear_component.sqrt()
    } else {
        0.
    }
}

#[inline]
fn reflect(v: Vec3A, normal: Vec3A) -> Vec3A {
    v - 2. * v.dot(normal) * normal
}

#[inline]
fn refract(uv: Vec3A, normal: Vec3A, ηi_over_ηt: f32) -> Vec3A {
    let cos_θ = (-uv).dot(normal).min(1.0);
    let r_out_perp = ηi_over_ηt * (uv + cos_θ * normal);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * normal;
    r_out_perp + r_out_parallel
}

fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
    let r0 = (1. - refraction_index) / (1. + refraction_index);
    let r0 = r0 * r0;
    r0 + (1. - r0) * (1. - cosine).powi(5)
}

#[inline]
fn random_in_unit_disc() -> Vec2 {
    loop {
        let p = 2.0 * Vec2::from(rand::random::<[f32; 2]>()) - Vec2::ONE;
        if p.length_squared() < 1. {
            return p
        }
    }
}
