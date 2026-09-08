# 스마트 카메라 카운터

휴대폰 카메라에 비치는 **사람**과 **차량**을 실시간으로 인식해서 개수를 세어주는 웹앱입니다.
설치 없이 브라우저(모바일 크롬/사파리)에서 바로 동작합니다.

## 동작 방식
- `TensorFlow.js` + `COCO-SSD` (lite_mobilenet_v2) 모델을 브라우저에서 실행
- 사람 = `person`, 차량 = `자전거 / 자동차 / 오토바이 / 버스 / 기차 / 트럭`
- 모든 처리가 휴대폰 안에서 이루어지고 영상은 서버로 전송되지 않습니다

## 파일
| 파일 | 설명 |
|------|------|
| `index.html` | 화면(UI) |
| `app.js` | 카메라 + AI 인식 로직 |
| `vercel.json` | 카메라 권한 헤더 설정 |

빌드 과정이 없는 순수 정적 사이트입니다.

## Vercel 배포 방법 (가장 쉬운 방법)

1. https://vercel.com 에 GitHub 계정으로 로그인
2. **Add New… → Project**
3. 이 폴더(`camera-vision-counter`)를 GitHub 저장소에 올린 뒤 선택
   - 또는 Vercel CLI 사용: 폴더에서 `npx vercel` 실행
4. Framework Preset은 **Other**, 나머지 기본값 그대로 **Deploy**
5. 발급된 `https://...vercel.app` 주소를 휴대폰에서 열기

> ⚠️ 카메라는 **HTTPS 주소(=vercel.app)** 에서만 켜집니다. 로컬 파일로 직접 열면 카메라가 동작하지 않습니다.

## 로컬 테스트
```
npx serve .
```
후 `http://localhost:3000` 접속 (localhost는 HTTPS 없이도 카메라 허용)

## 조정 옵션 (`app.js` 상단)
- `MIN_SCORE` : 인식 최소 확률 (기본 0.55). 오탐이 많으면 올리세요.
- `DETECT_EVERY_MS` : 인식 주기(ms). 값이 작을수록 반응 빠르지만 발열/버벅임 증가.
- `cocoSsd.load({ base: ... })` : `lite_mobilenet_v2`(빠름) → `mobilenet_v2`(정확) 로 바꿀 수 있음.
